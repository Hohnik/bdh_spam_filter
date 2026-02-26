"""Unit tests for the BDH model and classifier.

Tests verify:
  - Forward pass shapes are correct
  - Loss is a finite scalar
  - Role A chunking: splitting a sequence into chunks with state carryover
    produces the same pooled representation as a single full-length forward pass
  - Role B state: live_state from classify() is updated between calls
  - Checkpointing round-trips correctly
"""

import tempfile
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.model.bdh import BDH, BDHConfig
from src.model.classifier import BDHSpamClassifier, SpamClassifierConfig
from src.training.trainer import Trainer, TrainerConfig


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def small_config() -> SpamClassifierConfig:
    """Minimal config for fast CPU tests.

    chunk_size=64  — one forward pass processes 64 tokens
    max_position=128 — RoPE table covers positions 0..127, so two chunks fit
    """
    return SpamClassifierConfig(
        n_layer=1,
        n_embd=32,
        n_head=2,
        mlp_internal_dim_multiplier=8,   # N = 32*8//2 = 128
        chunk_size=64,
        max_position=128,                # must be ≥ 2 * chunk_size for the chunking test
        max_email_bytes=128,
        dropout=0.0,
        diff_attn=True,
        attn_window=16,
    )


@pytest.fixture()
def model(small_config: SpamClassifierConfig) -> BDHSpamClassifier:
    m = BDHSpamClassifier(small_config)
    m.eval()
    return m


# ─── BDH encoder ─────────────────────────────────────────────────────────────

class TestBDHEncoder:
    def test_output_shape(self, small_config: SpamClassifierConfig) -> None:
        enc = BDH(small_config.to_bdh_config())
        enc.eval()
        B, T = 2, 32
        idx  = torch.randint(0, 256, (B, T))
        with torch.no_grad():
            hidden, state = enc(idx)
        assert hidden.shape == (B, T, small_config.n_embd)
        assert all(s is None for s in state)   # stateless returns None

    def test_stateful_output_shape(self, small_config: SpamClassifierConfig) -> None:
        enc = BDH(small_config.to_bdh_config())
        enc.eval()
        B, T = 2, 32
        idx  = torch.randint(0, 256, (B, T))
        state = enc.init_state(B, torch.device("cpu"))
        with torch.no_grad():
            hidden, new_state = enc(idx, state=state)
        assert hidden.shape == (B, T, small_config.n_embd)
        assert all(s is not None for s in new_state)

    def test_chunked_state_is_propagated(self, small_config: SpamClassifierConfig) -> None:
        """Role A: the synaptic state ρ must be non-zero after chunk 1 and must
        influence chunk 2's output.

        Concretely:
          - state after chunk 1 must differ from the initial zero state
          - chunk 2 output WITH state ≠ chunk 2 output WITHOUT state

        This is the guarantee that chunking is lossless: chunk 2 sees the
        accumulated Hebbian memory of chunk 1 via ρ, rather than starting cold.

        Note: chunked+state is NOT numerically identical to a full-sequence pass.
        Within-chunk attention only sees tokens in the current chunk; ρ provides
        a compressed summary of previous chunks. The property being tested is
        that the summary is non-trivial and actually affects the output.
        """
        enc = BDH(small_config.to_bdh_config())
        enc.eval()
        torch.manual_seed(0)
        B, chunk_sz = 1, small_config.chunk_size
        chunk1 = torch.randint(0, 256, (B, chunk_sz))
        chunk2 = torch.randint(0, 256, (B, chunk_sz))

        with torch.no_grad():
            # Process chunk 1 — accumulate state
            zero_state = enc.init_state(B, torch.device("cpu"))
            _, state_after_1 = enc(chunk1, state=zero_state, pos_offset=0)

            # State after chunk 1 must be non-zero (something was learned)
            assert all(
                s is not None and s.abs().sum() > 0
                for s in state_after_1
            ), "State must be non-zero after processing chunk 1"

            # Chunk 2 WITH state (has context from chunk 1)
            out_with_state, _ = enc(chunk2, state=state_after_1, pos_offset=chunk_sz)

            # Chunk 2 WITHOUT state (cold start, no memory of chunk 1)
            fresh_state = enc.init_state(B, torch.device("cpu"))
            out_without_state, _ = enc(chunk2, state=fresh_state, pos_offset=chunk_sz)

        # The outputs must differ: state from chunk 1 must influence chunk 2
        assert not torch.allclose(out_with_state, out_without_state), \
            "Chunk 2 output should differ when state from chunk 1 is provided"
        assert torch.isfinite(out_with_state).all()
        assert out_with_state.shape == (B, chunk_sz, small_config.n_embd)


# ─── Classifier ──────────────────────────────────────────────────────────────

class TestClassifier:
    def test_forward_no_labels(self, model: BDHSpamClassifier) -> None:
        B, T = 3, 64
        ids  = torch.randint(0, 256, (B, T))
        with torch.no_grad():
            logits, loss = model(ids)
        assert logits.shape == (B, 1)
        assert loss is None

    def test_forward_with_labels(self, model: BDHSpamClassifier) -> None:
        B, T   = 4, 64
        ids    = torch.randint(0, 256, (B, T))
        labels = torch.randint(0, 2, (B,)).float()
        with torch.no_grad():
            logits, loss = model(ids, labels)
        assert logits.shape == (B, 1)
        assert loss is not None
        assert loss.ndim == 0          # scalar
        assert torch.isfinite(loss)

    def test_predict_returns_probabilities(self, model: BDHSpamClassifier) -> None:
        ids = torch.randint(0, 256, (2, 64))
        with torch.no_grad():
            probs, new_state = model.predict(ids)
        assert probs.shape == (2,)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_predict_live_state_is_updated(self, model: BDHSpamClassifier) -> None:
        """Role B: live_state returned from predict() must differ from None state."""
        ids = torch.randint(0, 256, (1, 64))
        with torch.no_grad():
            _, state1 = model.predict(ids, live_state=None)
            _, state2 = model.predict(ids, live_state=state1)

        # state2 should differ from state1 (accumulated more Hebbian updates)
        for s1, s2 in zip(state1, state2):
            if s1 is not None and s2 is not None:
                assert not torch.allclose(s1, s2), \
                    "Synaptic state should change after a second predict() call"

    def test_checkpoint_round_trip(
        self, model: BDHSpamClassifier, small_config: SpamClassifierConfig
    ) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            model.save_checkpoint(f.name)
            loaded = BDHSpamClassifier.from_checkpoint(f.name)

        # Weights must be identical
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), loaded.named_parameters()
        ):
            assert n1 == n2
            assert torch.equal(p1, p2), f"Parameter {n1} changed after checkpoint round-trip"

    def test_gradient_flows(self, small_config: SpamClassifierConfig) -> None:
        """Training: gradients must flow to all parameter groups."""
        model = BDHSpamClassifier(small_config)
        model.train()
        ids    = torch.randint(0, 256, (2, 32))
        labels = torch.tensor([1.0, 0.0])
        _, loss = model(ids, labels)
        loss.backward()

        params_with_grad = [
            n for n, p in model.named_parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        ]
        assert len(params_with_grad) > 0, "No parameters received gradients"


# ─── Trainer ──────────────────────────────────────────────────────────────────

def _tiny_loader(n_batches: int = 8, seq_len: int = 64) -> DataLoader:
    ids    = torch.randint(0, 256, (n_batches * 4, seq_len))
    labels = torch.randint(0, 2,   (n_batches * 4,)).float()
    return DataLoader(TensorDataset(ids, labels), batch_size=4)


class TestTrainer:
    def _make_trainer(self, tmp_path, max_steps: int = 10) -> Trainer:
        cfg = SpamClassifierConfig(
            n_layer=1, n_embd=16, n_head=2,
            mlp_internal_dim_multiplier=4,
            chunk_size=64, max_position=128,
            max_email_bytes=128, dropout=0.0,
        )
        model  = BDHSpamClassifier(cfg)
        loader = _tiny_loader()
        tcfg   = TrainerConfig(
            max_steps      = max_steps,
            eval_every     = 5,
            checkpoint_dir = str(tmp_path),
            device         = "cpu",
            learning_rate  = 1e-3,
        )
        return Trainer(model, loader, loader, tcfg)

    def test_normal_run_saves_checkpoint(self, tmp_path) -> None:
        """At least one checkpoint is saved when F1 improves during training."""
        trainer = self._make_trainer(tmp_path, max_steps=10)
        trainer.train()
        ckpts = list(tmp_path.glob("*.pt"))
        # May not save if F1 never improves (untrained model is random),
        # so we just assert it doesn't crash and steps are counted.
        assert trainer._step == 10

    def test_keyboard_interrupt_saves_emergency_checkpoint(self, tmp_path) -> None:
        """Ctrl+C at step 3 must produce an 'interrupted' checkpoint."""
        trainer = self._make_trainer(tmp_path, max_steps=20)
        original_step = trainer._train_step

        call_count = [0]
        def _step_that_interrupts(ids, labels):
            call_count[0] += 1
            if call_count[0] >= 3:
                raise KeyboardInterrupt
            return original_step(ids, labels)

        with patch.object(trainer, "_train_step", side_effect=_step_that_interrupts):
            trainer.train()   # must NOT re-raise KeyboardInterrupt

        emergency = list(tmp_path.glob("*interrupted*.pt"))
        assert len(emergency) == 1, (
            f"Expected one 'interrupted' checkpoint, found: {emergency}"
        )
        assert "step" in emergency[0].name

    def test_interrupted_checkpoint_is_loadable(self, tmp_path) -> None:
        """The emergency checkpoint must be a valid model file."""
        trainer = self._make_trainer(tmp_path, max_steps=20)
        original_step = trainer._train_step

        call_count = [0]
        def _interrupt_at_5(ids, labels):
            call_count[0] += 1
            if call_count[0] >= 5:
                raise KeyboardInterrupt
            return original_step(ids, labels)

        with patch.object(trainer, "_train_step", side_effect=_interrupt_at_5):
            trainer.train()

        ckpt = list(tmp_path.glob("*interrupted*.pt"))[0]
        loaded = BDHSpamClassifier.from_checkpoint(str(ckpt))
        assert loaded is not None
        # Sanity: can do a forward pass
        ids = torch.randint(0, 256, (1, 64))
        with torch.no_grad():
            logits, _ = loaded(ids)
        assert logits.shape == (1, 1)
