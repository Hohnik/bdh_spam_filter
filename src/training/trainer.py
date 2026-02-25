"""Initial (offline) training loop for the BDH spam classifier.

Trains on the full SpamAssassin + Enron corpus once. The resulting checkpoint
is the starting point for continuous deployment; SpamFilter.learn() handles
all subsequent updates.

Key design choices:
  - Metrics: F1, precision, recall — not just accuracy (corpus is imbalanced)
  - Cosine LR schedule with linear warmup
  - Gradient clipping (important for BDH's sparse ReLU gradients)
  - Checkpoint saved whenever validation F1 improves
"""

from __future__ import annotations

import dataclasses
import math
import time
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..model.classifier import BDHSpamClassifier


@dataclasses.dataclass
class TrainerConfig:
    # Optimiser
    learning_rate: float = 3e-4
    weight_decay:  float = 1e-2
    max_grad_norm: float = 1.0

    # Schedule: linear warmup → cosine decay
    warmup_steps: int = 200
    max_steps:    int = 3000

    # Evaluation
    eval_every:  int = 100
    checkpoint_dir: str = "checkpoints"

    # Device
    device: str = "cpu"   # "cpu" | "mps" | "cuda"


def _cosine_lr(step: int, cfg: TrainerConfig) -> float:
    """Learning rate multiplier for cosine schedule with linear warmup."""
    if step < cfg.warmup_steps:
        return step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def _evaluate(
    model: BDHSpamClassifier,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute loss, accuracy, precision, recall, and F1 on *loader*."""
    model.eval()
    total_loss = 0.0
    tp = fp = fn = tn = 0

    for token_ids, labels in loader:
        token_ids = token_ids.to(device)
        labels    = labels.to(device)
        logits, loss = model(token_ids, labels)
        if loss is not None:
            total_loss += loss.item() * token_ids.size(0)

        preds = (torch.sigmoid(logits.squeeze(1)) >= threshold).long()
        gt    = labels.long()
        tp += ((preds == 1) & (gt == 1)).sum().item()
        fp += ((preds == 1) & (gt == 0)).sum().item()
        fn += ((preds == 0) & (gt == 1)).sum().item()
        tn += ((preds == 0) & (gt == 0)).sum().item()

    n       = tp + fp + fn + tn
    acc     = (tp + tn) / n if n > 0 else 0.0
    prec    = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec     = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1      = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    avg_loss = total_loss / n if n > 0 else float("inf")
    return {"loss": avg_loss, "acc": acc, "precision": prec, "recall": rec, "f1": f1}


class Trainer:
    """Trains BDHSpamClassifier on an offline corpus of labelled emails.

    Usage:
        trainer = Trainer(model, train_loader, val_loader, config)
        trainer.train()
    """

    def __init__(
        self,
        model:        BDHSpamClassifier,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        config:       TrainerConfig,
    ) -> None:
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.config       = config
        self.device       = torch.device(config.device)

        self.model.to(self.device)

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: _cosine_lr(step, config),
        )

        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self._best_f1   = 0.0
        self._step      = 0

    # ── Training ──────────────────────────────────────────────────────────────

    def _train_step(self, token_ids: torch.Tensor, labels: torch.Tensor) -> float:
        self.model.train()
        token_ids = token_ids.to(self.device)
        labels    = labels.to(self.device)

        _, loss = self.model(token_ids, labels)
        if loss is None:
            raise RuntimeError("Loss is None — labels must be provided")

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    def train(self) -> None:
        """Run training for max_steps gradient updates."""
        cfg = self.config
        print(
            f"\nBDHSpamClassifier  ({self.model.parameter_count():,} params)  "
            f"on {cfg.device}"
        )

        train_iter = iter(self.train_loader)
        recent_losses: list[float] = []       # sliding window for smooth display

        bar = tqdm(
            total=cfg.max_steps,
            initial=self._step,
            unit="step",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                       "[{elapsed}<{remaining}, {rate_fmt}  {postfix}]",
        )

        while self._step < cfg.max_steps:
            try:
                token_ids, labels = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                token_ids, labels = next(train_iter)

            loss = self._train_step(token_ids, labels)
            self._step += 1

            recent_losses.append(loss)
            if len(recent_losses) > 20:
                recent_losses.pop(0)

            bar.update(1)
            bar.set_postfix(
                loss=f"{sum(recent_losses)/len(recent_losses):.4f}",
                lr=f"{self.scheduler.get_last_lr()[0]:.1e}",
            )

            # ── Periodic evaluation ──
            if self._step % cfg.eval_every == 0 or self._step == cfg.max_steps:
                metrics = _evaluate(self.model, self.val_loader, self.device)

                bar.set_postfix(
                    loss=f"{sum(recent_losses)/len(recent_losses):.4f}",
                    val_F1=f"{metrics['f1']:.4f}",
                    prec=f"{metrics['precision']:.4f}",
                    rec=f"{metrics['recall']:.4f}",
                    lr=f"{self.scheduler.get_last_lr()[0]:.1e}",
                )

                # Print a summary line below the bar (tqdm.write keeps bar intact)
                tqdm.write(
                    f"  step {self._step:>5d} | "
                    f"val_loss {metrics['loss']:.4f} | "
                    f"F1 {metrics['f1']:.4f} | "
                    f"prec {metrics['precision']:.4f} | "
                    f"rec {metrics['recall']:.4f}"
                )

                if metrics["f1"] > self._best_f1:
                    self._best_f1 = metrics["f1"]
                    ckpt_path = (
                        Path(cfg.checkpoint_dir)
                        / f"spam_step{self._step:05d}_f1{metrics['f1']:.4f}.pt"
                    )
                    self.model.save_checkpoint(str(ckpt_path))
                    tqdm.write(f"  ✓ new best F1 → {ckpt_path.name}")

        bar.close()
        print(f"\nDone.  Best validation F1: {self._best_f1:.4f}")
