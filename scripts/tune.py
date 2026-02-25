#!/usr/bin/env python3
"""Hyperparameter search with Optuna.

Searches the three parameters that are genuinely uncertain for spam classification.
The BDH structural params (n_layer=2, diff_attn=True, attn_window=64) are held
fixed — they are validated by extensive ablations in the parent repo.

Search space:
  learning_rate   log-uniform  [1e-4, 1e-3]
  n_embd          categorical  [64, 128, 256]
  dropout         uniform      [0.0, 0.25]

Trials use --chunk-size 128 (fast) by default.  The relative ranking of
lr / n_embd / dropout is stable across chunk sizes.  The final training run
uses --chunk-size 256 for better quality.

Usage:
    uv run python scripts/download_data.py     # only once
    uv run python scripts/tune.py --device mps --trials 25 --trial-steps 300

Best params are printed at the end and saved to checkpoints/best_hparams.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search")
    parser.add_argument("--device",       default="cpu",  help="cpu | mps | cuda")
    parser.add_argument("--trials",       type=int, default=25,  help="Optuna trials")
    parser.add_argument("--trial-steps",  type=int, default=300, help="Steps per trial")
    parser.add_argument("--chunk-size",   type=int, default=128,
                        help="chunk_size for tuning (128=fast, 256=quality)")
    parser.add_argument("--batch",        type=int, default=16,   help="Batch size")
    parser.add_argument("--data-dir",     default="data",         help="Dataset cache dir")
    parser.add_argument("--study-name",   default="bdh_spam",     help="Optuna study name")
    args = parser.parse_args()

    try:
        import optuna
    except ImportError:
        print("ERROR: optuna not installed.  Run:  uv add optuna")
        sys.exit(1)

    from src.data.download import download_all
    from src.data.dataset import build_dataloaders
    from src.model.classifier import SpamClassifierConfig
    from src.training.trainer import _evaluate

    # ── Load data ONCE (shared across all trials) ──────────────────────────────
    print("Loading dataset …")
    samples = download_all(Path(args.data_dir))

    # Use the smallest n_embd config to pre-build loaders at the right seq_len.
    # All trials share the same seq_len = chunk_size (different n_embd models
    # all accept the same token tensor shape).
    base_cfg     = SpamClassifierConfig(chunk_size=args.chunk_size)
    train_loader, val_loader = build_dataloaders(
        samples,
        seq_len         = base_cfg.chunk_size,
        max_email_bytes = base_cfg.max_email_bytes,
        batch_size      = args.batch,
    )
    print(f"Ready.  Device: {args.device.upper()}")

    device = torch.device(args.device)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # ── Objective ─────────────────────────────────────────────────────────────

    def objective(trial: Any) -> float:
        from src.model.classifier import BDHSpamClassifier, SpamClassifierConfig
        from src.training.trainer import TrainerConfig, Trainer

        lr      = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
        n_embd  = trial.suggest_categorical("n_embd", [64, 128, 256])
        dropout = trial.suggest_float("dropout", 0.0, 0.25)

        config = SpamClassifierConfig(
            n_embd   = n_embd,
            n_layer  = 2,
            n_head   = 4,
            dropout  = dropout,
            diff_attn    = True,
            attn_window  = 64,
            mlp_internal_dim_multiplier = 32,
            chunk_size   = args.chunk_size,
            max_position = 4096,
            max_email_bytes = 4096,
        )

        model = BDHSpamClassifier(config).to(device)
        opt   = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=1e-2, betas=(0.9, 0.95))

        train_iter = iter(train_loader)
        model.train()

        with tqdm(
            total=args.trial_steps,
            desc=(f"  trial {trial.number:>3d} "
                  f"lr={lr:.0e} embd={n_embd} drop={dropout:.2f}"),
            leave=False,
            ncols=78,
        ) as bar:
            for step in range(1, args.trial_steps + 1):
                try:
                    ids, lbl = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    ids, lbl   = next(train_iter)

                ids, lbl = ids.to(device), lbl.to(device)
                _, loss  = model(ids, lbl)
                opt.zero_grad(); loss.backward(); opt.step()
                bar.update(1)
                bar.set_postfix(loss=f"{loss.item():.4f}")

                # Median pruning: report at 50% progress
                if step == args.trial_steps // 2:
                    mid_metrics = _evaluate(model, val_loader, device)
                    trial.report(mid_metrics["f1"], step)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

        final = _evaluate(model, val_loader, device)
        return final["f1"]

    # ── Study ─────────────────────────────────────────────────────────────────
    study = optuna.create_study(
        direction  = "maximize",
        study_name = args.study_name,
        sampler    = optuna.samplers.TPESampler(seed=42),
        pruner     = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )

    print(f"\nSearch space:  lr=[1e-4,1e-3]  n_embd={{64,128,256}}  dropout=[0,0.25]")
    print(f"{args.trials} trials × {args.trial_steps} steps  chunk_size={args.chunk_size}\n")

    with tqdm(total=args.trials, desc="Trials", unit="trial", ncols=65) as study_bar:
        def _cb(study, trial):
            study_bar.update(1)
            study_bar.set_postfix(best_f1=f"{study.best_value:.4f}")

        study.optimize(objective, n_trials=args.trials, callbacks=[_cb])

    # ── Results ───────────────────────────────────────────────────────────────
    best = study.best_trial
    print(f"\n{'='*55}")
    print(f"  Best trial #{best.number}  —  F1: {best.value:.4f}")
    print(f"{'='*55}")
    for k, v in best.params.items():
        print(f"  {k:<20} {v}")

    out_dir  = Path("checkpoints")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "best_hparams.json"
    with open(out_path, "w") as f:
        json.dump({"f1": best.value, "params": best.params}, f, indent=2)
    print(f"\n  Saved to {out_path}")
    print(f"\n  Full training command:")
    lr_str  = f"{best.params['learning_rate']:.2e}"
    print(
        f"    uv run python scripts/train.py"
        f" --device {args.device}"
        f" --steps 5000"
        f" --lr {lr_str}"
        f" --embd {best.params['n_embd']}"
        f" --dropout {best.params['dropout']:.3f}"
    )


if __name__ == "__main__":
    main()
