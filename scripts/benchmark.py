#!/usr/bin/env python3
"""Model benchmark — no data download required.

Tests multiple (batch_size, chunk_size) combinations to find the fastest
training config for this machine.  MPS is tested with a hard timeout so the
script never hangs.

Run:
    uv run python scripts/benchmark.py
"""

from __future__ import annotations

import signal
import sys
import time
import tracemalloc
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple, Optional

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.classifier import BDHSpamClassifier, SpamClassifierConfig


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_batch(B: int, T: int, device: torch.device):
    ids    = torch.randint(0, 256, (B, T), device=device)
    labels = torch.randint(0, 2,   (B,),   device=device).float()
    return ids, labels


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


@contextmanager
def _timeout(seconds: int):
    """Raise TimeoutError if the block takes longer than *seconds*."""
    def _handler(sig, frame):
        raise TimeoutError
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


class StepResult(NamedTuple):
    device:        str
    batch_size:    int
    chunk_size:    int
    ms_per_step:   float
    ms_inference:  float   # batch=1
    emails_per_s:  float
    peak_ram_mb:   float


def _measure(
    device: torch.device,
    config: SpamClassifierConfig,
    batch_size: int,
    chunk_size: int,
    n_warmup: int = 5,
    n_measure: int = 10,
) -> Optional[StepResult]:
    """Benchmark one (device, B, T) combination; returns None on failure/timeout."""
    try:
        with _timeout(120):
            model = BDHSpamClassifier(config).to(device)
            opt   = torch.optim.AdamW(model.parameters(), lr=3e-4)

            model.train()
            for _ in range(n_warmup):
                ids, lbl = _make_batch(batch_size, chunk_size, device)
                _, loss  = model(ids, lbl); loss.backward()
                opt.step(); opt.zero_grad()
            _sync(device)

            # ── Inference latency (batch=1) ──
            model.eval()
            ids1, _ = _make_batch(1, chunk_size, device)
            _sync(device)
            t = time.perf_counter()
            for _ in range(20):
                with torch.no_grad():
                    model(ids1)
            _sync(device)
            ms_inf = (time.perf_counter() - t) / 20 * 1000

            # ── Training throughput ──
            model.train()
            tracemalloc.start()
            _sync(device)
            t0 = time.perf_counter()
            for _ in range(n_measure):
                ids, lbl = _make_batch(batch_size, chunk_size, device)
                _, loss  = model(ids, lbl); loss.backward()
                opt.step(); opt.zero_grad()
            _sync(device)
            elapsed = time.perf_counter() - t0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            ms   = elapsed / n_measure * 1000
            eps  = batch_size / (elapsed / n_measure)
            pmb  = peak / 1024**2
            return StepResult(str(device), batch_size, chunk_size, ms, ms_inf, eps, pmb)
    except (TimeoutError, RuntimeError, Exception):
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        return None


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # Base config (structural params fixed; B and T are swept below)
    base_config = SpamClassifierConfig(
        n_embd=128, n_layer=2, n_head=4,
        mlp_internal_dim_multiplier=32,
        dropout=0.1, diff_attn=True, attn_window=64,
        chunk_size=512, max_position=4096, max_email_bytes=4096,
    )

    model_probe  = BDHSpamClassifier(base_config)
    n_params     = model_probe.parameter_count()
    param_mb     = sum(p.numel() * p.element_size() for p in model_probe.parameters()) / 1024**2
    N            = base_config.n_embd * base_config.mlp_internal_dim_multiplier // base_config.n_head

    print("=" * 65)
    print("  BDH Spam Filter — Benchmark")
    print("=" * 65)
    print(f"\n  Architecture")
    print(f"  {'─'*40}")
    print(f"  Parameters       {n_params:>12,}")
    print(f"  Weights          {param_mb:>11.1f} MB  (fp32)")
    print(f"  n_embd / N       {base_config.n_embd:>4} / {N:<6}  (embed dim / sparse dim per head)")
    print(f"  n_layer / n_head {base_config.n_layer:>4} / {base_config.n_head:<6}")
    print(f"  diff_attn        yes   attn_window = {base_config.attn_window}")

    # ── Devices ───────────────────────────────────────────────────────────────
    devices: list[torch.device] = [torch.device("cpu")]
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    elif torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    # (batch_size, chunk_size) combos to sweep.
    # Attention is O(B * nh * T * T) so T dominates; sweep T first.
    configs_to_test: list[tuple[int, int]] = [
        (8,  128),
        (16, 128),
        (8,  256),
        (16, 256),
        (32, 256),
        (8,  512),
        (16, 512),
        (32, 512),
    ]

    print(f"\n  Speed sweep  (B = batch, T = chunk_size)")
    print(f"  {'─'*60}")
    print(f"  {'Device':<6}  {'B':>3}  {'T':>4}  {'ms/step':>8}  "
          f"{'inf ms':>7}  {'emails/s':>9}  {'RAM':>7}")
    print(f"  {'─'*60}")

    all_results: list[StepResult] = []

    total_runs = len(devices) * len(configs_to_test)
    with tqdm(total=total_runs, desc="  Sweeping", ncols=65, leave=True) as bar:
        for dev in devices:
            for B, T in configs_to_test:
                # Build a config with the right chunk_size
                cfg = SpamClassifierConfig(
                    n_embd=base_config.n_embd, n_layer=base_config.n_layer,
                    n_head=base_config.n_head,
                    mlp_internal_dim_multiplier=base_config.mlp_internal_dim_multiplier,
                    dropout=base_config.dropout, diff_attn=base_config.diff_attn,
                    attn_window=base_config.attn_window,
                    chunk_size=T, max_position=4096, max_email_bytes=4096,
                )
                r = _measure(dev, cfg, B, T)
                bar.update(1)
                if r is None:
                    tqdm.write(f"  {str(dev):<6}  {B:>3}  {T:>4}  {'TIMEOUT/ERR':>8}")
                else:
                    all_results.append(r)
                    tqdm.write(
                        f"  {r.device:<6}  {r.batch_size:>3}  {r.chunk_size:>4}  "
                        f"{r.ms_per_step:>7.0f}ms  "
                        f"{r.ms_inference:>6.1f}ms  "
                        f"{r.emails_per_s:>8.0f}/s  "
                        f"{r.peak_ram_mb:>5.0f}MB"
                    )

    if not all_results:
        print("\n  ERROR: all benchmarks failed.")
        return

    # ── Best config for training ───────────────────────────────────────────────
    # "Quality-best" = largest (B * T) per second = most informative bytes/s.
    # This differs from fastest emails/s: T=256 sees the email body, T=128 only
    # the subject.  For spam detection that needs to catch body-level phishing
    # T=256 is the right minimum.
    def _bytes_per_sec(r: StepResult) -> float:
        return r.batch_size * r.chunk_size / (r.ms_per_step / 1000)

    # Pick best (device-specific) among T>=256 configs; fall back to any if none
    quality_results = [r for r in all_results if r.chunk_size >= 256]
    pool = quality_results if quality_results else all_results
    best = max(pool, key=_bytes_per_sec)

    # Corpus estimate: SpamAssassin ~10,800 + Enron ~33,000 = ~43,800 emails
    CORPUS  = 43_800
    STEPS_5K  = 5_000
    STEPS_10K = 10_000

    secs_5k  = STEPS_5K  * best.ms_per_step / 1000
    secs_10k = STEPS_10K * best.ms_per_step / 1000
    steps_per_epoch = CORPUS // best.batch_size

    print(f"\n  Best config:  device={best.device}  B={best.batch_size}  T={best.chunk_size}")
    print(f"  {'─'*55}")
    print(f"  {best.ms_per_step:.0f} ms/step  →  "
          f"{best.emails_per_s:.0f} emails/s training  |  "
          f"{best.ms_inference:.1f} ms inference")
    print(f"  Steps / epoch (corpus={CORPUS:,}):  {steps_per_epoch:,}")
    print(f"  5 000 steps  ≈  {secs_5k/60:.0f} min")
    print(f"  10 000 steps ≈  {secs_10k/60:.0f} min")

    # ── Device comparison ──────────────────────────────────────────────────────
    cpu_results = [r for r in all_results if r.device == "cpu"]
    mps_results = [r for r in all_results if r.device == "mps"]
    if cpu_results and mps_results:
        best_cpu = max(cpu_results, key=lambda r: r.emails_per_s)
        best_mps = max(mps_results, key=lambda r: r.emails_per_s)
        speedup  = best_mps.emails_per_s / best_cpu.emails_per_s
        if speedup > 1.15:
            rec_device = best_mps.device
            print(f"\n  MPS is {speedup:.1f}× faster than CPU for the best config — use MPS.")
        elif speedup < 0.85:
            rec_device = best_cpu.device
            print(f"\n  CPU is {1/speedup:.1f}× faster than MPS for this model size — use CPU.")
        else:
            rec_device = best_cpu.device
            print(f"\n  MPS/CPU are comparable (±15%) — use CPU (simpler, more stable).")
    else:
        rec_device = best.device

    # ── Recommendations ───────────────────────────────────────────────────────
    print(f"\n  {'='*63}")
    print("  Recommendations")
    print(f"  {'='*63}")

    rec_B = best.batch_size
    rec_T = best.chunk_size
    mins_5k  = secs_5k / 60
    mins_10k = secs_10k / 60

    print(f"""
  Training command (recommended settings):

    uv run python scripts/train.py \\
        --device {rec_device} \\
        --batch  {rec_B} \\
        --steps  {STEPS_5K if mins_5k <= 30 else 3000}

  Duration:
    5 000 steps  ≈  {mins_5k:.0f} min  ← {"start here" if mins_5k <= 30 else "may be long — try 3 000 first"}
    10 000 steps ≈  {mins_10k:.0f} min  ← fully saturated

  chunk_size = {rec_T} bytes/chunk.  Context guide:
    T=128  → subject line only.  Fast (2× speedup), but misses body phishing.
    T=256  → subject + first paragraph.  Best quality/speed trade-off (default).
    T=512  → full short emails.  2-4× slower; rarely needed for spam detection.
  The RoPE table covers positions 0–{base_config.max_position}, so chunking
  handles long newsletters without discarding any content.
""")

    print(f"  Hyperparameter tuning (Optuna)")
    print(f"  {'─'*40}")
    n_trials    = 25
    trial_steps = 300

    # Optuna trials use T=128 (4× faster than T=256). The relative ranking of
    # lr / n_embd / dropout is stable across chunk sizes — only throughput changes.
    fast_results = [r for r in all_results
                    if r.device == rec_device and r.chunk_size == 128]
    tune_ms      = (max(fast_results, key=lambda r: r.batch_size).ms_per_step
                    if fast_results else best.ms_per_step)
    tune_mins    = n_trials * trial_steps * tune_ms / 1000 / 60

    print(f"""
  3 params to search: learning_rate, n_embd, dropout
  (n_layer=2, diff_attn=True, attn_window=64 are fixed — validated by
   extensive ablations in the parent baby_dragon_hatchling repo)

  Optuna uses T=128 for speed; final training uses T=256 for quality.
  {n_trials} trials × {trial_steps} steps  ≈  {tune_mins:.0f} min on {rec_device.upper()}

  {"RECOMMENDED — fast enough to be worth it" if tune_mins < 45 else "OPTIONAL — run if you have time; defaults are reasonable"}

  Run tuning first (fast T=128):
    uv run python scripts/tune.py --device {rec_device} --trials {n_trials} --trial-steps {trial_steps}

  Then train with the best params (quality T=256):
    uv run python scripts/train.py --device {rec_device} --steps 5000 \\
        --lr <best_lr> --embd <best_embd> --dropout <best_dropout>
""")
    print(f"  {'='*63}\n")


if __name__ == "__main__":
    main()
