"""SpamDataset and DataLoader factory.

Each sample is a fixed-length integer tensor of byte values (0-255) and a
binary float label (1.0 = spam, 0.0 = ham).

Padding value: 0 (null byte — rare in real emails, so it's a safe pad token).
"""

from __future__ import annotations

import random
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from .download import Sample
from .email_parser import extract_text, text_to_token_ids


class SpamDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch Dataset over a list of (raw_email_text, label) pairs.

    Token IDs are padded / truncated to *seq_len* bytes. The same seq_len is
    used for every sample so DataLoader can form batches without collation.
    """

    PAD_ID = 0

    def __init__(
        self,
        samples: Sequence[Sample],
        seq_len: int,
        max_email_bytes: int = 4096,
    ) -> None:
        self.seq_len         = seq_len
        self.max_email_bytes = max_email_bytes
        self._tokens: list[list[int]] = []
        self._labels: list[float]     = []

        for sample in samples:
            text   = extract_text(sample.text)
            ids    = text_to_token_ids(text, max_bytes=max_email_bytes)
            self._tokens.append(ids)
            self._labels.append(float(sample.label))

    def __len__(self) -> int:
        return len(self._tokens)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids = self._tokens[idx]
        # Pad or truncate to seq_len
        if len(ids) < self.seq_len:
            ids = ids + [self.PAD_ID] * (self.seq_len - len(ids))
        else:
            ids = ids[: self.seq_len]
        token_tensor = torch.tensor(ids, dtype=torch.long)
        label_tensor = torch.tensor(self._labels[idx], dtype=torch.float32)
        return token_tensor, label_tensor


def build_dataloaders(
    samples: Sequence[Sample],
    seq_len: int,
    max_email_bytes: int = 4096,
    batch_size: int = 32,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Split *samples* into train/val and return two DataLoaders.

    Stratified split: maintains the spam/ham ratio in both halves.

    Returns:
        train_loader, val_loader
    """
    rng    = random.Random(seed)
    spam   = [s for s in samples if s.label == 1]
    ham    = [s for s in samples if s.label == 0]

    def split(lst: list[Sample]) -> tuple[list[Sample], list[Sample]]:
        lst = lst[:]
        rng.shuffle(lst)
        n_val = max(1, int(len(lst) * val_fraction))
        return lst[n_val:], lst[:n_val]

    spam_train, spam_val = split(spam)
    ham_train,  ham_val  = split(ham)

    train_samples = spam_train + ham_train
    val_samples   = spam_val  + ham_val
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    train_ds = SpamDataset(train_samples, seq_len, max_email_bytes)
    val_ds   = SpamDataset(val_samples,   seq_len, max_email_bytes)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,   # 0 = main process; avoids issues on macOS / small servers
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    print(
        f"Dataset: {len(train_ds)} train  /  {len(val_ds)} val  "
        f"(batch_size={batch_size}, seq_len={seq_len})"
    )
    return train_loader, val_loader
