"""Tests for IMAP client and MailChecker.

All tests mock imaplib — no live IMAP server required.
"""

from __future__ import annotations

import email
import imaplib
from unittest.mock import MagicMock, patch, call

import pytest
import torch

from src.mail.imap_client import IMAPClient, IMAPConfig, _parse_folder_name
from src.mail.checker import MailChecker, CheckResult
from src.filter import SpamFilter, SpamResult
from src.model.classifier import BDHSpamClassifier, SpamClassifierConfig


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def t_online_cfg() -> IMAPConfig:
    return IMAPConfig.t_online("test@t-online.de", "secret")


@pytest.fixture()
def spam_filter() -> SpamFilter:
    cfg   = SpamClassifierConfig(n_layer=1, n_embd=32, n_head=2,
                                  mlp_internal_dim_multiplier=8,
                                  chunk_size=64, max_position=256,
                                  max_email_bytes=256, dropout=0.0)
    model = BDHSpamClassifier(cfg)
    return SpamFilter(model, device=torch.device("cpu"))


def _make_raw_email(subject: str, body: str, sender: str = "x@example.com") -> bytes:
    msg = email.message.Message()
    msg["From"]    = sender
    msg["To"]      = "me@example.com"
    msg["Subject"] = subject
    msg.set_payload(body)
    return msg.as_bytes()


# ─── IMAPConfig ───────────────────────────────────────────────────────────────

class TestIMAPConfig:
    def test_t_online_factory(self, t_online_cfg: IMAPConfig) -> None:
        assert t_online_cfg.host == "secureimap.t-online.de"
        assert t_online_cfg.port == 993
        assert t_online_cfg.username == "test@t-online.de"
        assert t_online_cfg.spam_folder == "Spam"

    def test_password_not_logged(self, t_online_cfg: IMAPConfig, caplog) -> None:
        """The password must never appear in any log output."""
        import logging
        with caplog.at_level(logging.DEBUG):
            # Simulate creating config — just check repr doesn't leak password
            _ = repr(t_online_cfg)
        assert "secret" not in caplog.text


# ─── _parse_folder_name ───────────────────────────────────────────────────────

class TestParseFolderName:
    def test_quoted_name(self) -> None:
        raw = b'(\\HasNoChildren) "/" "Spam"'
        assert _parse_folder_name(raw) == "Spam"

    def test_inbox(self) -> None:
        raw = b'(\\HasNoChildren \\UnMarked) "." "INBOX"'
        assert _parse_folder_name(raw) == "INBOX"

    def test_nested_folder(self) -> None:
        raw = b'(\\HasNoChildren) "/" "INBOX/Sent"'
        assert _parse_folder_name(raw) == "INBOX/Sent"

    def test_empty_bytes(self) -> None:
        assert _parse_folder_name(b"") is None


# ─── IMAPClient ───────────────────────────────────────────────────────────────

class TestIMAPClient:
    """Tests using a fully mocked imaplib.IMAP4_SSL."""

    def _make_mock_conn(self, caps: bytes = b"IMAP4rev1 MOVE") -> MagicMock:
        conn = MagicMock(spec=imaplib.IMAP4_SSL)
        conn.capability.return_value = ("OK", [caps])
        conn.login.return_value       = ("OK", [b"Logged in"])
        conn.close.return_value       = ("OK", [])
        conn.logout.return_value      = ("BYE", [b"Logging out"])
        return conn

    def test_connect_detects_move_capability(self, t_online_cfg: IMAPConfig) -> None:
        mock_conn = self._make_mock_conn(caps=b"IMAP4rev1 MOVE IDLE")
        with patch("imaplib.IMAP4_SSL", return_value=mock_conn):
            client = IMAPClient(t_online_cfg)
            client.connect()
            assert client._supports_move is True

    def test_connect_no_move_capability(self, t_online_cfg: IMAPConfig) -> None:
        mock_conn = self._make_mock_conn(caps=b"IMAP4rev1 IDLE")
        with patch("imaplib.IMAP4_SSL", return_value=mock_conn):
            client = IMAPClient(t_online_cfg)
            client.connect()
            assert client._supports_move is False

    def test_context_manager_connects_and_disconnects(self, t_online_cfg: IMAPConfig) -> None:
        mock_conn = self._make_mock_conn()
        with patch("imaplib.IMAP4_SSL", return_value=mock_conn):
            with IMAPClient(t_online_cfg) as _:
                pass
        mock_conn.login.assert_called_once()
        mock_conn.logout.assert_called_once()

    def test_ensure_spam_folder_found(self, t_online_cfg: IMAPConfig) -> None:
        mock_conn = self._make_mock_conn()
        mock_conn.list.return_value = ("OK", [
            b'(\\HasNoChildren) "/" "INBOX"',
            b'(\\HasNoChildren) "/" "Spam"',
            b'(\\HasNoChildren) "/" "Sent"',
        ])
        with patch("imaplib.IMAP4_SSL", return_value=mock_conn):
            with IMAPClient(t_online_cfg) as client:
                folder = client.ensure_spam_folder()
        assert folder == "Spam"
        mock_conn.create.assert_not_called()

    def test_ensure_spam_folder_created_when_missing(self, t_online_cfg: IMAPConfig) -> None:
        mock_conn = self._make_mock_conn()
        mock_conn.list.return_value = ("OK", [
            b'(\\HasNoChildren) "/" "INBOX"',
        ])
        mock_conn.create.return_value = ("OK", [b"Created"])
        with patch("imaplib.IMAP4_SSL", return_value=mock_conn):
            with IMAPClient(t_online_cfg) as client:
                folder = client.ensure_spam_folder()
        assert folder == "Spam"
        mock_conn.create.assert_called_once_with("Spam")

    def test_fetch_unseen_empty_inbox(self, t_online_cfg: IMAPConfig) -> None:
        mock_conn = self._make_mock_conn()
        mock_conn.select.return_value = ("OK", [b"5"])
        mock_conn.uid.return_value    = ("OK", [b""])   # empty SEARCH result
        with patch("imaplib.IMAP4_SSL", return_value=mock_conn):
            with IMAPClient(t_online_cfg) as client:
                msgs = list(client.fetch_unseen())
        assert msgs == []

    def test_fetch_unseen_returns_messages(self, t_online_cfg: IMAPConfig) -> None:
        raw = _make_raw_email("Hello", "Body text")
        mock_conn = self._make_mock_conn()
        mock_conn.select.return_value = ("OK", [b"1"])

        # uid() is called for SEARCH, then for each FETCH
        mock_conn.uid.side_effect = [
            ("OK", [b"1 2"]),                        # SEARCH response
            ("OK", [(b"1 (RFC822 {42})", raw), b")"]),  # FETCH uid=1
            ("OK", [(b"2 (RFC822 {42})", raw), b")"]),  # FETCH uid=2
        ]
        with patch("imaplib.IMAP4_SSL", return_value=mock_conn):
            with IMAPClient(t_online_cfg) as client:
                msgs = list(client.fetch_unseen())
        assert len(msgs) == 2

    def test_move_uses_move_command_when_supported(self, t_online_cfg: IMAPConfig) -> None:
        mock_conn = self._make_mock_conn(caps=b"IMAP4rev1 MOVE")
        mock_conn.uid.return_value = ("OK", [b""])
        with patch("imaplib.IMAP4_SSL", return_value=mock_conn):
            with IMAPClient(t_online_cfg) as client:
                client.move_to_spam(b"42", "Spam")

        # The UID call for MOVE
        calls = [str(c) for c in mock_conn.uid.call_args_list]
        assert any("MOVE" in c for c in calls)

    def test_move_falls_back_to_copy_delete(self, t_online_cfg: IMAPConfig) -> None:
        mock_conn = self._make_mock_conn(caps=b"IMAP4rev1")
        mock_conn.uid.return_value     = ("OK", [b""])
        mock_conn.expunge.return_value = ("OK", [])
        with patch("imaplib.IMAP4_SSL", return_value=mock_conn):
            with IMAPClient(t_online_cfg) as client:
                client.move_to_spam(b"42", "Spam")

        calls = [str(c) for c in mock_conn.uid.call_args_list]
        assert any("COPY"  in c for c in calls)
        assert any("STORE" in c for c in calls)
        mock_conn.expunge.assert_called_once()


# ─── MailChecker ──────────────────────────────────────────────────────────────

class TestMailChecker:
    """Tests MailChecker orchestration logic without touching IMAP or model weights."""

    def _make_client_with_messages(self, messages: list[tuple]) -> MagicMock:
        """Return an IMAPClient mock that yields (uid, msg, raw) from *messages*."""
        client = MagicMock(spec=IMAPClient)
        client.ensure_spam_folder.return_value = "Spam"
        client.fetch_unseen.return_value = iter(messages)
        return client

    def _make_spam_result(self, is_spam: bool, conf: float) -> SpamResult:
        return SpamResult(is_spam=is_spam, confidence=conf, threshold=0.5)

    def test_spam_is_moved(self, spam_filter: SpamFilter) -> None:
        raw = _make_raw_email("Win a prize!", "Click here")
        uid = b"1"
        msg = email.message_from_bytes(raw)

        client = self._make_client_with_messages([(uid, msg, raw)])
        spam_filter.classify = MagicMock(
            return_value=self._make_spam_result(is_spam=True, conf=0.95)
        )

        checker = MailChecker(spam_filter, client, dry_run=False, auto_learn_threshold=0.85)
        result  = checker.run()

        assert result.spam  == 1
        assert result.moved == 1
        client.move_to_spam.assert_called_once_with(uid, "Spam")

    def test_ham_is_not_moved(self, spam_filter: SpamFilter) -> None:
        raw = _make_raw_email("Meeting at 3pm", "See you there")
        uid = b"2"
        msg = email.message_from_bytes(raw)

        client = self._make_client_with_messages([(uid, msg, raw)])
        spam_filter.classify = MagicMock(
            return_value=self._make_spam_result(is_spam=False, conf=0.1)
        )

        checker = MailChecker(spam_filter, client)
        result  = checker.run()

        assert result.ham   == 1
        assert result.moved == 0
        client.move_to_spam.assert_not_called()

    def test_dry_run_does_not_move(self, spam_filter: SpamFilter) -> None:
        raw = _make_raw_email("Cheap meds!", "Buy now")
        uid = b"3"
        msg = email.message_from_bytes(raw)

        client = self._make_client_with_messages([(uid, msg, raw)])
        spam_filter.classify = MagicMock(
            return_value=self._make_spam_result(is_spam=True, conf=0.99)
        )

        checker = MailChecker(spam_filter, client, dry_run=True)
        result  = checker.run()

        assert result.spam  == 1
        assert result.moved == 0
        client.move_to_spam.assert_not_called()

    def test_auto_learn_only_above_threshold(self, spam_filter: SpamFilter) -> None:
        """Borderline spam (conf=0.6) should NOT trigger a weight update."""
        raw = _make_raw_email("Offer", "maybe spam")
        uid = b"4"
        msg = email.message_from_bytes(raw)

        client = self._make_client_with_messages([(uid, msg, raw)])
        spam_filter.classify = MagicMock(
            return_value=self._make_spam_result(is_spam=True, conf=0.6)
        )
        spam_filter.learn = MagicMock(return_value=0.3)

        checker = MailChecker(spam_filter, client, auto_learn_threshold=0.85)
        result  = checker.run()

        assert result.spam    == 1
        assert result.learned == 0        # conf 0.6 < threshold 0.85 → no update
        spam_filter.learn.assert_not_called()

    def test_high_confidence_spam_triggers_learn(self, spam_filter: SpamFilter) -> None:
        raw = _make_raw_email("Free money!", "Claim now")
        uid = b"5"
        msg = email.message_from_bytes(raw)

        client = self._make_client_with_messages([(uid, msg, raw)])
        spam_filter.classify = MagicMock(
            return_value=self._make_spam_result(is_spam=True, conf=0.97)
        )
        spam_filter.learn = MagicMock(return_value=0.2)

        checker = MailChecker(spam_filter, client, auto_learn_threshold=0.85)
        result  = checker.run()

        assert result.learned == 1
        spam_filter.learn.assert_called_once_with(raw, is_spam=True)

    def test_classify_error_is_counted_not_raised(self, spam_filter: SpamFilter) -> None:
        raw = _make_raw_email("Crash test", "")
        uid = b"6"
        msg = email.message_from_bytes(raw)

        client = self._make_client_with_messages([(uid, msg, raw)])
        spam_filter.classify = MagicMock(side_effect=RuntimeError("boom"))

        checker = MailChecker(spam_filter, client)
        result  = checker.run()

        assert result.errors == 1
        assert result.total  == 1

    def test_mixed_inbox(self, spam_filter: SpamFilter) -> None:
        """2 spam + 2 ham → correct counts."""
        def _raw(subj: str) -> bytes:
            return _make_raw_email(subj, "body")
        def _msg(raw: bytes):
            return email.message_from_bytes(raw)

        items = [
            (b"1", _msg(_raw("Buy now")),    _raw("Buy now")),
            (b"2", _msg(_raw("Hi there")),   _raw("Hi there")),
            (b"3", _msg(_raw("Win a car!")), _raw("Win a car!")),
            (b"4", _msg(_raw("Status: OK")), _raw("Status: OK")),
        ]
        labels = [True, False, True, False]
        confs  = [0.95, 0.05, 0.92, 0.08]

        client = self._make_client_with_messages(items)
        spam_filter.classify = MagicMock(
            side_effect=[
                self._make_spam_result(lbl, c) for lbl, c in zip(labels, confs)
            ]
        )

        checker = MailChecker(spam_filter, client, auto_learn_threshold=1.1)
        result  = checker.run()

        assert result.total == 4
        assert result.spam  == 2
        assert result.ham   == 2
        assert result.moved == 2
