"""Integration tests for the SpamFilter public interface.

Tests the full pipeline: email → filter → SpamResult.
No network I/O — uses a freshly-initialised (untrained) model.
An untrained model produces random predictions, so we only assert on
shapes, types, and stateful behaviour — not on correctness.
"""

import tempfile

import pytest
import torch

from src.filter import SpamFilter, SpamResult
from src.model.classifier import BDHSpamClassifier, SpamClassifierConfig


SPAM_EMAIL = """\
From: prize@spam.net
Subject: Congratulations! Claim your $500 gift card NOW
Content-Type: text/plain

Click the link to claim your free prize immediately!
"""

HAM_EMAIL = """\
From: boss@company.com
Subject: Q3 report review

Hi team, please review the attached Q3 report before Friday.
Thanks.
"""


@pytest.fixture()
def filter_instance() -> SpamFilter:
    """A SpamFilter backed by a tiny untrained model (fast, CPU)."""
    config = SpamClassifierConfig(
        n_layer=1,
        n_embd=32,
        n_head=2,
        mlp_internal_dim_multiplier=8,
        chunk_size=64,
        max_position=256,
        max_email_bytes=256,
        dropout=0.0,
    )
    model = BDHSpamClassifier(config)
    return SpamFilter(model, device=torch.device("cpu"))


class TestSpamFilterClassify:
    def test_returns_spam_result(self, filter_instance: SpamFilter) -> None:
        result = filter_instance.classify(SPAM_EMAIL)
        assert isinstance(result, SpamResult)

    def test_confidence_in_unit_interval(self, filter_instance: SpamFilter) -> None:
        for email in (SPAM_EMAIL, HAM_EMAIL):
            result = filter_instance.classify(email)
            assert 0.0 <= result.confidence <= 1.0

    def test_is_spam_consistent_with_threshold(self, filter_instance: SpamFilter) -> None:
        result = filter_instance.classify(SPAM_EMAIL)
        expected = result.confidence >= result.threshold
        assert result.is_spam == expected

    def test_bytes_input_accepted(self, filter_instance: SpamFilter) -> None:
        result = filter_instance.classify(SPAM_EMAIL.encode("utf-8"))
        assert isinstance(result, SpamResult)

    def test_custom_threshold(self, filter_instance: SpamFilter) -> None:
        filter_instance.threshold = 0.0   # everything is spam
        result = filter_instance.classify(HAM_EMAIL)
        assert result.is_spam is True

        filter_instance.threshold = 1.0   # nothing is spam
        result = filter_instance.classify(HAM_EMAIL)
        assert result.is_spam is False

        filter_instance.threshold = SpamFilter.DEFAULT_THRESHOLD  # restore


class TestSpamFilterLiveState:
    def test_state_updates_between_calls(self, filter_instance: SpamFilter) -> None:
        """Role B: classifying two emails should leave a non-None live state."""
        assert filter_instance._live_state is None   # fresh start
        filter_instance.classify(SPAM_EMAIL)
        assert filter_instance._live_state is not None

    def test_reset_state_clears(self, filter_instance: SpamFilter) -> None:
        filter_instance.classify(SPAM_EMAIL)
        filter_instance.reset_state()
        assert filter_instance._live_state is None

    def test_repeated_calls_differ_in_state(self, filter_instance: SpamFilter) -> None:
        """Each classify() call should update ρ, so state after 2 calls ≠ after 1."""
        filter_instance.classify(SPAM_EMAIL)
        state_after_1 = [s.clone() for s in filter_instance._live_state]

        filter_instance.classify(HAM_EMAIL)
        state_after_2 = filter_instance._live_state

        any_changed = any(
            not torch.allclose(s1, s2)
            for s1, s2 in zip(state_after_1, state_after_2)
            if s1 is not None and s2 is not None
        )
        assert any_changed, "Live state should change after each classify() call"


class TestSpamFilterLearn:
    def test_learn_returns_finite_loss(self, filter_instance: SpamFilter) -> None:
        loss = filter_instance.learn(SPAM_EMAIL, is_spam=True)
        assert isinstance(loss, float)
        assert 0.0 < loss < 1e6   # BCE loss is always positive and finite

    def test_learn_does_not_corrupt_classify(self, filter_instance: SpamFilter) -> None:
        """After a learn() call, classify() must still return a valid SpamResult."""
        filter_instance.learn(SPAM_EMAIL, is_spam=True)
        filter_instance.learn(HAM_EMAIL,  is_spam=False)
        result = filter_instance.classify(SPAM_EMAIL)
        assert isinstance(result, SpamResult)
        assert 0.0 <= result.confidence <= 1.0


class TestSpamFilterPersistence:
    def test_save_and_load_round_trip(self, filter_instance: SpamFilter) -> None:
        # Build up some live state, then save at a known point.
        filter_instance.classify(SPAM_EMAIL)
        # Save BEFORE classifying HAM so both original and loaded start from
        # the identical state when they process HAM_EMAIL below.
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            fname = f.name
        filter_instance.save(fname)

        # Both should produce the same confidence because they start from the
        # same saved state and process the same email.
        original_conf = filter_instance.classify(HAM_EMAIL).confidence
        loaded        = SpamFilter.load(fname)
        loaded_conf   = loaded.classify(HAM_EMAIL).confidence

        assert abs(original_conf - loaded_conf) < 1e-5

    def test_load_restores_live_state(self, filter_instance: SpamFilter) -> None:
        filter_instance.classify(SPAM_EMAIL)   # populate state
        state_before = [s.clone() for s in filter_instance._live_state]

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            filter_instance.save(f.name)
            loaded = SpamFilter.load(f.name)

        assert loaded._live_state is not None
        for s_orig, s_load in zip(state_before, loaded._live_state):
            if s_orig is not None and s_load is not None:
                assert torch.allclose(s_orig, s_load, atol=1e-6), \
                    "Live state should survive a save/load round-trip"
