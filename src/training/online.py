"""OnlineLearner — continuous learning for deployed spam filter.

This implements Role B of the BDH recurrent state: cross-session learning.

When a user marks an email as spam or ham, the model performs a single gradient
step on that example. The key challenges for online learning are:

  1. Catastrophic forgetting: a large LR update on one new example can destroy
     previously learned patterns. Mitigated by:
       - Very small learning rate (orders of magnitude below training LR)
       - Gradient clipping
       - Replay buffer: mix the new example with a small random sample of past
         examples so each update also rehearses old patterns

  2. Class imbalance drift: users report many more spam than ham. The replay
     buffer maintains a balanced spam/ham ratio to prevent the model drifting
     towards always predicting spam.

  3. State persistence: the BDH synaptic state ρ is updated during inference
     (Role B). This is separate from the gradient-based weight updates here.
     Both mechanisms work together — weights capture slow structural patterns,
     ρ captures fast session-level patterns.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Optional

import torch
from torch import nn
from torch.optim import AdamW

from ..model.classifier import BDHSpamClassifier
from ..data.email_parser import extract_text, text_to_token_ids


class OnlineLearner:
    """Performs incremental gradient updates on user-confirmed spam/ham examples.

    Maintains an in-memory replay buffer (separate spam and ham queues) to
    prevent catastrophic forgetting. Every learn() call does one gradient step
    on a batch containing the new example + replayed past examples.

    Args:
        model:         The deployed BDHSpamClassifier (modified in-place).
        lr:            Online learning rate. Keep much smaller than training LR.
                       Default 1e-5 (training used 3e-4 — 30× smaller).
        max_grad_norm: Gradient clipping threshold.
        buffer_size:   Max examples per class stored in the replay buffer.
        replay_k:      Number of past examples to mix into each update step.
                       Higher = more stable but more memory accesses.
        device:        Torch device for gradient updates.
    """

    def __init__(
        self,
        model:         BDHSpamClassifier,
        lr:            float = 1e-5,
        max_grad_norm: float = 0.5,
        buffer_size:   int   = 256,
        replay_k:      int   = 4,
        device:        Optional[torch.device] = None,
    ) -> None:
        self.model         = model
        self.max_grad_norm = max_grad_norm
        self.replay_k      = replay_k
        self.device        = device or next(model.parameters()).device

        # Separate queues per class for balanced replay
        self._spam_buf: deque[list[int]] = deque(maxlen=buffer_size)
        self._ham_buf:  deque[list[int]] = deque(maxlen=buffer_size)

        # Use a smaller learning rate and independent optimizer state so online
        # updates don't interfere with any residual Adam state from training.
        self._optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=0.0,   # no decay for online updates
            betas=(0.9, 0.999),
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def learn(self, raw_email: str | bytes, is_spam: bool) -> float:
        """Incorporate one confirmed spam/ham example into the model.

        Args:
            raw_email: Raw RFC-2822 email text or bytes.
            is_spam:   True if the email is spam, False if ham.

        Returns:
            loss: BCE loss for this update step (for logging).
        """
        token_ids = self._preprocess(raw_email)
        label     = 1.0 if is_spam else 0.0

        # Add to replay buffer BEFORE the update (so it can be replayed later)
        if is_spam:
            self._spam_buf.append(token_ids)
        else:
            self._ham_buf.append(token_ids)

        # Build a mini-batch: new example + replay
        batch_ids, batch_labels = self._build_batch(token_ids, label)

        # One gradient step
        loss = self._gradient_step(batch_ids, batch_labels)
        return loss

    def buffer_stats(self) -> dict[str, int]:
        return {"spam_buffer": len(self._spam_buf), "ham_buffer": len(self._ham_buf)}

    # ── Internals ─────────────────────────────────────────────────────────────

    def _preprocess(self, raw_email: str | bytes) -> list[int]:
        """Parse email and convert to token IDs."""
        if isinstance(raw_email, bytes):
            raw_email = raw_email.decode("utf-8", errors="replace")
        text = extract_text(raw_email)
        return text_to_token_ids(text, max_bytes=self.model.config.max_email_bytes)

    def _build_batch(
        self, new_ids: list[int], new_label: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Assemble a training mini-batch from the new example + replay."""
        seq_len     = self.model.config.chunk_size
        all_ids:    list[list[int]] = [new_ids]
        all_labels: list[float]     = [new_label]

        # Balanced replay: draw replay_k // 2 from each class
        k_each = max(1, self.replay_k // 2)
        for buf, lbl in ((self._spam_buf, 1.0), (self._ham_buf, 0.0)):
            if buf:
                sampled = random.choices(list(buf), k=min(k_each, len(buf)))
                all_ids.extend(sampled)
                all_labels.extend([lbl] * len(sampled))

        def _pad(ids: list[int]) -> list[int]:
            if len(ids) < seq_len:
                return ids + [0] * (seq_len - len(ids))
            return ids[:seq_len]

        token_tensor = torch.tensor(
            [_pad(ids) for ids in all_ids], dtype=torch.long, device=self.device
        )
        label_tensor = torch.tensor(all_labels, dtype=torch.float32, device=self.device)
        return token_tensor, label_tensor

    def _gradient_step(
        self, token_ids: torch.Tensor, labels: torch.Tensor
    ) -> float:
        """Perform a single forward+backward+update step."""
        self.model.train()
        _, loss = self.model(token_ids, labels)
        if loss is None:
            raise RuntimeError("Loss is None")

        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self._optimizer.step()
        return loss.item()
