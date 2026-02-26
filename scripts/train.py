#!/usr/bin/env python3
"""Train the BDH spam classifier from scratch.

    uv run python scripts/train.py [--steps N] [--device cpu|mps|cuda]

The script:
  1. Downloads SpamAssassin + Enron corpora (cached locally after first run)
  2. Builds train/val DataLoaders with stratified split
  3. Trains BDHSpamClassifier for --steps gradient updates
  4. Saves the best checkpoint (by val F1) to checkpoints/

On a typical home server (no GPU, modern CPU):
  ~3.2M parameter model; run scripts/benchmark.py for accurate time estimates.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.data.download import download_all
from src.data.dataset import build_dataloaders
from src.model.classifier import BDHSpamClassifier, SpamClassifierConfig
from src.training.trainer import Trainer, TrainerConfig


def _preflight_check(model_config: "SpamClassifierConfig", args: argparse.Namespace) -> None:
    """Run 5 warm-up steps, project total training time, abort if too slow.

    Runs BEFORE downloading 41 K emails so a bad config fails in seconds not
    minutes.  Prints a clear recommendation and asks the user to confirm if
    the projected time exceeds the warning threshold.
    """
    import time

    WARN_MINUTES   = 90    # warn if projected > 90 min
    ABORT_MINUTES  = 300   # auto-abort if projected > 5 hours
    WARMUP_STEPS   = 5

    device_str = args.device
    if device_str == "mps" and not torch.backends.mps.is_available():
        device_str = "cpu"
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"

    device = torch.device(device_str)
    model  = BDHSpamClassifier(model_config).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    T   = model_config.chunk_size
    B   = args.batch

    # Warm-up then time
    print(f"\nPre-flight: {WARMUP_STEPS} steps @ B={B}, T={T}, "
          f"n_embd={model_config.n_embd}, "
          f"mlp_mult={model_config.mlp_internal_dim_multiplier} …", flush=True)

    for _ in range(2):   # un-timed warm-up
        ids    = torch.randint(0, 256, (B, T), device=device)
        labels = torch.randint(0, 2,   (B,),   device=device).float()
        _, loss = model(ids, labels)
        opt.zero_grad(); loss.backward(); opt.step()

    t0 = time.perf_counter()
    for _ in range(WARMUP_STEPS):
        ids    = torch.randint(0, 256, (B, T), device=device)
        labels = torch.randint(0, 2,   (B,),   device=device).float()
        _, loss = model(ids, labels)
        opt.zero_grad(); loss.backward(); opt.step()
        if device_str == "mps":
            torch.mps.synchronize()
        elif device_str == "cuda":
            torch.cuda.synchronize()
    elapsed   = time.perf_counter() - t0

    del model, opt
    if device_str == "mps":
        torch.mps.empty_cache()

    ms_per_step  = elapsed / WARMUP_STEPS * 1000
    total_min    = ms_per_step * args.steps / 1000 / 60
    params       = BDHSpamClassifier(model_config).parameter_count()

    print(f"  {ms_per_step:>7.0f} ms/step  →  "
          f"{total_min:.0f} min for {args.steps} steps  "
          f"({params:,} params)")

    if total_min > ABORT_MINUTES:
        N = model_config.n_embd * model_config.mlp_internal_dim_multiplier // 4
        good_mult = max(8, 1024 * 4 // model_config.n_embd)
        good_N    = model_config.n_embd * good_mult // 4
        print(
            f"\n✗  Too slow to run ({total_min:.0f} min > {ABORT_MINUTES} min limit).\n"
            f"\n   Root cause:  n_embd={model_config.n_embd} × mlp_mult="
            f"{model_config.mlp_internal_dim_multiplier} → N={N} MLP units/head.\n"
            f"   Benchmark baseline is N=1024 (mlp_mult={good_mult} for this n_embd).\n"
            f"\n   Recommended fix — add this flag:\n"
            f"     --mlp-mult {good_mult}   (N={good_N}, same throughput as benchmark)\n"
            f"\n   Or reduce model size:\n"
            f"     --embd 128   (3.2M params, {ms_per_step/4:.0f} ms/step estimated)\n"
        )
        sys.exit(1)

    if total_min > WARN_MINUTES:
        N = model_config.n_embd * model_config.mlp_internal_dim_multiplier // 4
        good_mult = max(8, 1024 * 4 // model_config.n_embd)
        print(
            f"\n⚠  Slow config ({total_min:.0f} min). N={N} MLP units/head "
            f"vs benchmark N=1024.\n"
            f"   To speed up:  add --mlp-mult {good_mult}  "
            f"(saves ~{total_min - total_min*1024/N:.0f} min)\n"
        )
        try:
            ans = input("   Continue anyway? [y/N] ").strip().lower()
        except EOFError:
            ans = "n"
        if ans != "y":
            sys.exit(0)

    print()   # blank line before data loading output


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BDH spam filter")
    parser.add_argument("--steps",    type=int,   default=5000,  help="Training steps")
    parser.add_argument("--device",   type=str,   default="cpu", help="cpu | mps | cuda")
    parser.add_argument("--lr",       type=float, default=3e-4,  help="Learning rate")
    parser.add_argument("--batch",    type=int,   default=32,    help="Batch size")
    parser.add_argument("--embd",     type=int,   default=128,   help="n_embd (model width)")
    parser.add_argument("--dropout",  type=float, default=0.1,   help="Dropout rate")
    parser.add_argument("--mlp-mult", type=int,   default=None,
                        help="mlp_internal_dim_multiplier. Default: auto-scaled so "
                             "MLP internal dim N = 1024 * n_head / n_embd (keeps "
                             "throughput constant regardless of --embd).")
    parser.add_argument("--data-dir",  default="data",           help="Dataset cache dir")
    parser.add_argument("--ckpt-dir",  default="checkpoints",    help="Checkpoint output dir")
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    # mlp_internal_dim_multiplier: keep MLP internal dim N ≈ 1024 regardless of
    # n_embd so that throughput stays close to the benchmarked baseline.
    #   n_embd=64  → multiplier=64 (N=64*64/4=1024)
    #   n_embd=128 → multiplier=32 (N=128*32/4=1024)  ← benchmark baseline
    #   n_embd=256 → multiplier=16 (N=256*16/4=1024)
    # Users can override with --mlp-mult but are warned if N > 1024.
    N_HEAD     = 4
    TARGET_N   = 1024   # MLP units per head at the benchmark sweet spot
    auto_mult  = max(8, TARGET_N * N_HEAD // args.embd)
    mlp_mult   = args.mlp_mult if args.mlp_mult is not None else auto_mult

    computed_N = args.embd * mlp_mult // N_HEAD
    if computed_N > TARGET_N:
        print(
            f"\n⚠  WARNING: MLP internal dim N = {computed_N} (n_embd={args.embd} × "
            f"mlp_mult={mlp_mult} / n_head={N_HEAD}).\n"
            f"   Benchmark was tuned for N={TARGET_N}. Larger N exponentially "
            f"increases RAM and compute.\n"
            f"   To keep benchmark speed, omit --mlp-mult (auto = {auto_mult}).\n"
        )

    model_config = SpamClassifierConfig(
        n_embd                      = args.embd,
        dropout                     = args.dropout,
        mlp_internal_dim_multiplier = mlp_mult,
    )

    # ── Pre-flight speed check ─────────────────────────────────────────────────
    # Time 3 steps before loading 41K emails. Abort early if it's too slow.
    _preflight_check(model_config, args)

    # ── Data ──────────────────────────────────────────────────────────────────
    samples = download_all(Path(args.data_dir))
    train_loader, val_loader = build_dataloaders(
        samples,
        seq_len         = model_config.chunk_size,
        max_email_bytes = model_config.max_email_bytes,
        batch_size      = args.batch,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = BDHSpamClassifier(model_config)
    print(f"Model parameters: {model.parameter_count():,}")

    # ── Device ────────────────────────────────────────────────────────────────
    device_str = args.device
    if device_str == "mps" and not torch.backends.mps.is_available():
        print("MPS not available — falling back to CPU")
        device_str = "cpu"
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available — falling back to CPU")
        device_str = "cpu"

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer_cfg = TrainerConfig(
        learning_rate  = args.lr,
        max_steps      = args.steps,
        warmup_steps   = max(100, args.steps // 10),
        eval_every     = max(50, args.steps // 30),
        checkpoint_dir = args.ckpt_dir,
        device         = device_str,
    )
    trainer = Trainer(model, train_loader, val_loader, trainer_cfg)
    trainer.train()


if __name__ == "__main__":
    main()
