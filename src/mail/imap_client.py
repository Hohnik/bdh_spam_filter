"""Low-level IMAP client for secure inbox access.

Uses only stdlib (imaplib, ssl, email) — no extra dependencies.

Protocol decisions:
  - UID-based operations throughout: sequence numbers change as messages are
    fetched/deleted; UIDs are stable for the lifetime of a session.
  - MOVE (RFC 6851) if the server advertises it; COPY+STORE+EXPUNGE fallback.
  - SSL via ssl.create_default_context() — validates the server certificate.

T-Online settings (pre-configured in IMAPConfig.t_online()):
  Host:  secureimap.t-online.de
  Port:  993
  SSL:   yes (IMAP4_SSL)
"""

from __future__ import annotations

import dataclasses
import email
import email.header
import imaplib
import logging
import re
import ssl
from email.message import Message
from typing import Iterator

log = logging.getLogger(__name__)


@dataclasses.dataclass
class IMAPConfig:
    host:         str
    port:         int
    username:     str
    password:     str
    inbox_folder: str = "INBOX"
    spam_folder:  str = "Spam"

    @classmethod
    def t_online(cls, username: str, password: str) -> "IMAPConfig":
        """Pre-configured factory for T-Online / Telekom Mail."""
        return cls(
            host     = "secureimap.t-online.de",
            port     = 993,
            username = username,
            password = password,
            spam_folder = "Spam",
        )


def _decode_header(raw: str) -> str:
    """Decode a potentially RFC-2047-encoded header value to plain text."""
    try:
        parts = email.header.decode_header(raw)
        decoded = email.header.make_header(parts)
        return str(decoded)
    except Exception:
        return raw


def _parse_folder_name(imap_list_item: bytes) -> str | None:
    """Extract the mailbox name from an IMAP LIST response line.

    LIST response format: b'(\\HasNoChildren) "/" "Folder Name"'
    The folder name may or may not be quoted; the delimiter varies by server.
    """
    try:
        text = imap_list_item.decode("utf-8", errors="replace")
        # Split off the flags and delimiter; take everything after the last space-delim
        # Use regex to find the last quoted-or-unquoted token
        m = re.search(r'"([^"]+)"\s*$', text)
        if m:
            return m.group(1)
        # Unquoted name (no spaces)
        m = re.search(r'"\s*([^\s"]+)\s*$', text)
        if m:
            return m.group(1)
    except Exception:
        pass
    return None


class IMAPClient:
    """Context-managed IMAP client.

    Usage:
        with IMAPClient(config) as client:
            spam_folder = client.ensure_spam_folder()
            for uid, msg in client.fetch_unseen():
                ...
                client.move_to_spam(uid, spam_folder)
    """

    def __init__(self, config: IMAPConfig) -> None:
        self._cfg  = config
        self._conn: imaplib.IMAP4_SSL | None = None
        self._supports_move = False

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "IMAPClient":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def connect(self) -> None:
        log.info("Connecting to %s:%d …", self._cfg.host, self._cfg.port)
        ctx = ssl.create_default_context()
        self._conn = imaplib.IMAP4_SSL(self._cfg.host, self._cfg.port, ssl_context=ctx)
        self._conn.login(self._cfg.username, self._cfg.password)
        log.info("Authenticated as %s", self._cfg.username)

        # Probe server capabilities
        _, caps_data = self._conn.capability()
        caps = (caps_data[0] or b"").decode("ascii", errors="replace").upper()
        self._supports_move = "MOVE" in caps
        log.debug("Server capabilities: %s", caps)
        log.debug("RFC 6851 MOVE supported: %s", self._supports_move)

    def disconnect(self) -> None:
        if self._conn is None:
            return
        try:
            self._conn.close()
        except Exception:
            pass
        try:
            self._conn.logout()
        except Exception:
            pass
        self._conn = None
        log.info("Disconnected from %s", self._cfg.host)

    # ── Folder operations ─────────────────────────────────────────────────────

    def list_folders(self) -> list[str]:
        """Return all folder names visible to the authenticated user."""
        assert self._conn is not None
        _, raw_list = self._conn.list()
        folders: list[str] = []
        for item in raw_list or []:
            if isinstance(item, bytes):
                name = _parse_folder_name(item)
                if name:
                    folders.append(name)
        return folders

    def ensure_spam_folder(self) -> str:
        """Return the spam folder name, creating it if absent.

        Checks common names in priority order:
          configured spam_folder → Spam → Junk → Junk Mail → INBOX.Spam
        """
        assert self._conn is not None
        existing = self.list_folders()
        lower_map = {f.lower(): f for f in existing}

        candidates = [
            self._cfg.spam_folder,
            "Spam", "SPAM", "spam",
            "Junk", "JUNK", "junk",
            "Junk Mail", "Junk E-Mail",
            "INBOX.Spam", "INBOX/Spam",
        ]
        for candidate in candidates:
            if candidate in existing:
                log.debug("Using existing spam folder: %s", candidate)
                return candidate
            if candidate.lower() in lower_map:
                found = lower_map[candidate.lower()]
                log.debug("Using existing spam folder (case-matched): %s", found)
                return found

        # Create the configured spam folder
        log.info("Spam folder '%s' not found — creating it.", self._cfg.spam_folder)
        self._conn.create(self._cfg.spam_folder)
        return self._cfg.spam_folder

    # ── Message operations ────────────────────────────────────────────────────

    def fetch_unseen(
        self, folder: str | None = None
    ) -> Iterator[tuple[bytes, Message, bytes]]:
        """Yield (uid, parsed_message, raw_bytes) for every unseen email.

        Args:
            folder: Mailbox to search; defaults to config.inbox_folder.

        Yields each unseen message exactly once per call.  Messages are NOT
        marked as seen here — the caller decides whether to mark them.
        """
        assert self._conn is not None
        folder = folder or self._cfg.inbox_folder
        self._conn.select(f'"{folder}"')

        _, uid_data = self._conn.uid("SEARCH", None, "UNSEEN")
        if not uid_data or not uid_data[0]:
            log.info("No unseen messages in %s.", folder)
            return

        uids: list[bytes] = uid_data[0].split()
        log.info("%d unseen message(s) in %s.", len(uids), folder)

        for uid in uids:
            try:
                _, msg_data = self._conn.uid("FETCH", uid, "(RFC822)")
                if not msg_data or not msg_data[0]:
                    log.warning("Empty fetch for UID %s — skipping.", uid.decode())
                    continue
                raw: bytes = msg_data[0][1]  # type: ignore[index]
                msg = email.message_from_bytes(raw)
                yield uid, msg, raw
            except Exception as exc:
                log.error("Failed to fetch UID %s: %s", uid.decode(), exc)

    def move_to_spam(self, uid: bytes, spam_folder: str) -> None:
        """Move a message to the spam folder.

        Uses UID MOVE (RFC 6851) if the server supports it — atomic, no
        expunge needed.  Falls back to COPY + STORE(\\Deleted) + EXPUNGE.
        """
        assert self._conn is not None
        folder_arg = f'"{spam_folder}"'
        if self._supports_move:
            status, _ = self._conn.uid("MOVE", uid, folder_arg)
            if status == "OK":
                log.debug("UID %s moved to %s (MOVE)", uid.decode(), spam_folder)
                return
            # Server advertised MOVE but it failed — fall through to manual method
            log.warning("MOVE command failed (status=%s); falling back.", status)

        # Fallback: COPY → mark Deleted → EXPUNGE
        self._conn.uid("COPY", uid, folder_arg)
        self._conn.uid("STORE", uid, "+FLAGS", r"(\Deleted)")
        self._conn.expunge()
        log.debug("UID %s copied to %s, original deleted (COPY+DELETE)", uid.decode(), spam_folder)

    def mark_seen(self, uid: bytes) -> None:
        """Mark a message as \\Seen (read)."""
        assert self._conn is not None
        self._conn.uid("STORE", uid, "+FLAGS", r"(\Seen)")
