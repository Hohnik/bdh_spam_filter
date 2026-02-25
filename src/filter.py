"""SpamFilter — the single public entry point for the deployed spam classifier.

Typical usage:

    # First run: train from scratch
    from src.filter import SpamFilter
    sf = SpamFilter.train_new(checkpoint_dir="checkpoints")

    # Subsequent runs: load from checkpoint
    sf = SpamFilter.load("checkpoints/best.pt")

    # Classification
    result = sf.classify(raw_email_bytes)
    print(result.is_spam, result.confidence)

    # Online correction (user marks as spam/ham)
    sf.learn(raw_email_bytes, is_spam=True)

    # Persist updated model + state
    sf.save("checkpoints/updated.pt")

Two mechanisms keep the filter accurate over time:

  Gradient updates (OnlineLearner.learn)
    Weight-level learning: slow, persistent, generalises across future emails.
    Small LR with replay buffer prevents catastrophic forgetting.

  BDH synaptic state ρ (Role B)
    Session-level Hebbian memory: fast, resets on load unless saved explicitly.
    Tracks patterns within the current inference session.
    Saved/loaded alongside model weights via save()/load().
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

import torch

from .model.classifier import BDHSpamClassifier, SpamClassifierConfig
from .data.email_parser import extract_text, text_to_token_ids
from .training.online import OnlineLearner


@dataclasses.dataclass
class SpamResult:
    """Result of a single email classification."""
    is_spam:    bool
    confidence: float   # P(spam) in [0, 1]
    threshold:  float   # decision boundary used


class SpamFilter:
    """Deployed spam filter: classify emails and continuously learn from corrections.

    Holds:
      - model:   BDHSpamClassifier (weights, architecture config)
      - state:   BDH synaptic state ρ for the current session (Role B)
      - learner: OnlineLearner for weight-level updates on user corrections
    """

    DEFAULT_THRESHOLD = 0.5

    def __init__(
        self,
        model:     BDHSpamClassifier,
        device:    torch.device,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self.model     = model.to(device)
        self.device    = device
        self.threshold = threshold

        # Role B: persistent synaptic state across inference calls.
        # Initialised to zeros; accumulates as emails are classified.
        self._live_state: Optional[list[torch.Tensor]] = None

        # Online learner for weight-level updates
        self._learner = OnlineLearner(model, device=device)

    # ── Classification ────────────────────────────────────────────────────────

    def classify(self, raw_email: str | bytes) -> SpamResult:
        """Classify one email.

        The BDH synaptic state ρ is updated during this call (Role B) — the
        model accumulates context from every email it processes in this session.

        Args:
            raw_email: Raw RFC-2822 email (text or bytes).

        Returns:
            SpamResult with is_spam, confidence, and the decision threshold used.
        """
        token_ids = self._preprocess(raw_email)
        with torch.no_grad():
            probs, self._live_state = self.model.predict(
                token_ids, live_state=self._live_state
            )
        confidence = probs[0].item()
        return SpamResult(
            is_spam    = confidence >= self.threshold,
            confidence = confidence,
            threshold  = self.threshold,
        )

    # ── Continuous learning ───────────────────────────────────────────────────

    def learn(self, raw_email: str | bytes, is_spam: bool) -> float:
        """Incorporate a user correction into the model weights.

        This is a gradient-based weight update (small LR, replay buffer).
        It is *separate* from the synaptic state update that happens during
        classify() — both mechanisms run in parallel during deployment.

        Args:
            raw_email: The email the user is correcting.
            is_spam:   True = user says this is spam; False = not spam (ham).

        Returns:
            BCE loss for this update step (useful for logging).
        """
        loss = self._learner.learn(raw_email, is_spam)
        return loss

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save model weights, config, and the current synaptic state ρ to disk.

        Saving ρ means the continuous learning session can be resumed without
        losing the accumulated Hebbian memory.
        """
        state_to_save = None
        if self._live_state is not None:
            state_to_save = [s.cpu() if s is not None else None for s in self._live_state]

        torch.save(
            {
                "config":      dataclasses.asdict(self.model.config),
                "model_state": self.model.state_dict(),
                "live_state":  state_to_save,
                "threshold":   self.threshold,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str,
        device: Optional[torch.device] = None,
        threshold: Optional[float] = None,
    ) -> "SpamFilter":
        """Load a SpamFilter from a checkpoint saved with save().

        Restores both model weights and the saved synaptic state ρ, so the
        continuous learning session picks up exactly where it left off.
        """
        if device is None:
            device = torch.device("cpu")

        ckpt   = torch.load(path, map_location=device, weights_only=False)
        config = SpamClassifierConfig(**ckpt["config"])
        model  = BDHSpamClassifier(config)
        model.load_state_dict(ckpt["model_state"])

        sf = cls(
            model,
            device    = device,
            threshold = threshold if threshold is not None else ckpt.get("threshold", cls.DEFAULT_THRESHOLD),
        )
        # Restore synaptic state if present
        raw_state = ckpt.get("live_state")
        if raw_state is not None:
            sf._live_state = [
                s.to(device) if s is not None else None for s in raw_state
            ]
        return sf

    def reset_state(self) -> None:
        """Discard the current synaptic state ρ (start a fresh session)."""
        self._live_state = None

    # ── Factory: train from scratch ───────────────────────────────────────────

    @classmethod
    def train_new(
        cls,
        checkpoint_dir: str = "checkpoints",
        data_dir:       str = "data",
        device_str:     str = "cpu",
        config:         Optional[SpamClassifierConfig] = None,
    ) -> "SpamFilter":
        """Download training data, train, and return a ready-to-use SpamFilter.

        This is the all-in-one entry point for first-time setup.

        Args:
            checkpoint_dir: Where to save model checkpoints.
            data_dir:       Where to cache downloaded datasets.
            device_str:     "cpu", "mps", or "cuda".
            config:         Override default SpamClassifierConfig.
        """
        from pathlib import Path as _Path
        from .data.download import download_all
        from .data.dataset import build_dataloaders
        from .training.trainer import Trainer, TrainerConfig

        device = torch.device(device_str)

        if config is None:
            config = SpamClassifierConfig()

        # ── Data ──
        samples = download_all(_Path(data_dir))
        train_loader, val_loader = build_dataloaders(
            samples,
            seq_len         = config.chunk_size,
            max_email_bytes = config.max_email_bytes,
            batch_size      = 32,
        )

        # ── Model ──
        model = BDHSpamClassifier(config)
        print(f"\nModel: {model.parameter_count():,} parameters")

        # ── Train ──
        trainer_cfg = TrainerConfig(
            learning_rate   = 3e-4,
            max_steps       = 3000,
            warmup_steps    = 300,
            eval_every      = 100,
            checkpoint_dir  = checkpoint_dir,
            device          = device_str,
        )
        trainer = Trainer(model, train_loader, val_loader, trainer_cfg)
        trainer.train()

        sf = cls(model, device=device)
        best_path = str(_Path(checkpoint_dir) / "best.pt")
        sf.save(best_path)
        print(f"\nFilter ready. Saved to {best_path}")
        return sf

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _preprocess(self, raw_email: str | bytes) -> torch.Tensor:
        """Parse → text → byte IDs → (1, T) tensor on device.

        T is the actual email length (up to max_email_bytes), NOT chunk_size.
        _encode() takes care of splitting T into chunk_size-sized pieces.
        Padding is applied only when T < chunk_size (single-chunk case).
        """
        if isinstance(raw_email, bytes):
            raw_email = raw_email.decode("utf-8", errors="replace")
        text     = extract_text(raw_email)
        ids      = text_to_token_ids(text, max_bytes=self.model.config.max_email_bytes)
        # Ensure at least chunk_size tokens so a single forward pass always works
        chunk_sz = self.model.config.chunk_size
        if len(ids) < chunk_sz:
            ids = ids + [0] * (chunk_sz - len(ids))
        return torch.tensor([ids], dtype=torch.long, device=self.device)
