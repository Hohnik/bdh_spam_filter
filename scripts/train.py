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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BDH spam filter")
    parser.add_argument("--steps",   type=int,   default=5000,  help="Training steps")
    parser.add_argument("--device",  type=str,   default="cpu", help="cpu | mps | cuda")
    parser.add_argument("--lr",      type=float, default=3e-4,  help="Learning rate")
    parser.add_argument("--batch",   type=int,   default=32,    help="Batch size")
    parser.add_argument("--embd",    type=int,   default=128,   help="n_embd (model width)")
    parser.add_argument("--dropout", type=float, default=0.1,   help="Dropout rate")
    parser.add_argument("--data-dir",  default="data",          help="Dataset cache dir")
    parser.add_argument("--ckpt-dir",  default="checkpoints",   help="Checkpoint output dir")
    args = parser.parse_args()

    # ── Data ──────────────────────────────────────────────────────────────────
    samples = download_all(Path(args.data_dir))

    model_config = SpamClassifierConfig(
        n_embd   = args.embd,
        dropout  = args.dropout,
    )
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
