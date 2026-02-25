"""Unit tests for email parsing and dataset construction.

No network I/O — all tests use synthetic emails.
"""

import pytest
import torch

from src.data.email_parser import extract_text, text_to_token_ids
from src.data.dataset import SpamDataset
from src.data.download import Sample


# ─── Synthetic email fixtures ─────────────────────────────────────────────────

PLAIN_EMAIL = """\
From: spammer@evil.com
To: victim@example.com
Subject: You have won $1,000,000!!!
Date: Mon, 01 Jan 2024 00:00:00 +0000
MIME-Version: 1.0
Content-Type: text/plain; charset=utf-8

Congratulations! Click here to claim your prize.
http://totally-legit.ru/claim
"""

HTML_EMAIL = """\
From: newsletter@legit.com
To: user@example.com
Subject: Weekly update
Content-Type: text/html; charset=utf-8

<html><body><h1>Hello!</h1><p>Here is your <b>weekly</b> update.</p></body></html>
"""

MULTIPART_EMAIL = """\
From: sender@example.com
To: recipient@example.com
Subject: Multipart test
MIME-Version: 1.0
Content-Type: multipart/mixed; boundary="BOUNDARY"

--BOUNDARY
Content-Type: text/plain; charset=utf-8

Plain text part.

--BOUNDARY
Content-Type: text/html; charset=utf-8

<p>HTML part</p>

--BOUNDARY
Content-Type: application/pdf

%PDF binary garbage that should be ignored

--BOUNDARY--
"""


# ─── Email parser ─────────────────────────────────────────────────────────────

class TestExtractText:
    def test_plain_email_contains_subject(self) -> None:
        text = extract_text(PLAIN_EMAIL)
        assert "Subject" in text
        assert "won" in text.lower() or "1,000,000" in text

    def test_html_email_strips_tags(self) -> None:
        text = extract_text(HTML_EMAIL)
        assert "<" not in text, "HTML tags should be stripped"
        assert "weekly" in text.lower()

    def test_multipart_keeps_text_drops_binary(self) -> None:
        text = extract_text(MULTIPART_EMAIL)
        assert "Plain text part" in text
        assert "HTML part" in text
        assert "%PDF" not in text, "Binary PDF attachment should be dropped"

    def test_bytes_input_accepted(self) -> None:
        text = extract_text(PLAIN_EMAIL.encode("utf-8"))
        assert isinstance(text, str)
        assert len(text) > 0

    def test_garbage_input_does_not_crash(self) -> None:
        text = extract_text(b"\x00\xff\xfe malformed \r\n email \x80")
        assert isinstance(text, str)


class TestTextToTokenIds:
    def test_length_respects_max_bytes(self) -> None:
        long_text = "A" * 10_000
        ids = text_to_token_ids(long_text, max_bytes=512)
        assert len(ids) <= 512

    def test_all_values_in_byte_range(self) -> None:
        ids = text_to_token_ids("Hello, world! 🌍", max_bytes=256)
        assert all(0 <= v <= 255 for v in ids)

    def test_empty_string_returns_empty(self) -> None:
        ids = text_to_token_ids("", max_bytes=256)
        assert ids == []


# ─── Dataset ─────────────────────────────────────────────────────────────────

class TestSpamDataset:
    @pytest.fixture()
    def samples(self) -> list[Sample]:
        return [
            Sample(text=PLAIN_EMAIL, label=1),
            Sample(text=HTML_EMAIL,  label=0),
            Sample(text=MULTIPART_EMAIL, label=0),
        ]

    def test_length(self, samples: list[Sample]) -> None:
        ds = SpamDataset(samples, seq_len=64)
        assert len(ds) == len(samples)

    def test_item_shapes(self, samples: list[Sample]) -> None:
        seq_len = 64
        ds = SpamDataset(samples, seq_len=seq_len)
        token_ids, label = ds[0]
        assert token_ids.shape == (seq_len,)
        assert token_ids.dtype == torch.long
        assert label.shape == ()
        assert label.dtype == torch.float32

    def test_label_values(self, samples: list[Sample]) -> None:
        ds = SpamDataset(samples, seq_len=64)
        _, label0 = ds[0]
        _, label1 = ds[1]
        assert label0.item() == 1.0   # PLAIN_EMAIL is spam
        assert label1.item() == 0.0   # HTML_EMAIL is ham

    def test_padding_applied_to_short_emails(self) -> None:
        short = Sample(text="Subject: Hi\n\nHello", label=0)
        ds    = SpamDataset([short], seq_len=512)
        ids, _ = ds[0]
        assert ids.shape == (512,)
        # Trailing tokens should be the pad value (0)
        assert ids[-1].item() == SpamDataset.PAD_ID

    def test_token_values_in_range(self, samples: list[Sample]) -> None:
        ds = SpamDataset(samples, seq_len=128)
        for i in range(len(ds)):
            ids, label = ds[i]
            assert (ids >= 0).all() and (ids <= 255).all()
            assert label.item() in (0.0, 1.0)
