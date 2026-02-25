"""Extract clean, classifier-ready text from raw RFC-2822 email messages.

The BDH classifier is byte-level (vocab_size=256) so no tokeniser is needed.
This module's job is purely to flatten the MIME structure into a single UTF-8
string and then encode it as a list of byte values in [0, 255].

What we keep:
  - Subject header (highly discriminative)
  - From / Reply-To (sender address pattern)
  - Body text (text/plain parts, or a stripped version of text/html)

What we discard:
  - Binary attachments (images, PDFs)
  - Inline base64 blobs that would contribute random bytes
  - Redundant headers (Message-ID, MIME-Version, Content-Transfer-Encoding …)
"""

from __future__ import annotations

import email
import email.policy
import html
import re
from email.message import Message


# Tags to strip from HTML bodies (very basic, stdlib only — no BeautifulSoup dep)
_TAG_RE      = re.compile(r"<[^>]+>")
_WHITESPACE  = re.compile(r"\s+")
_HEADER_KEEP = ("from", "reply-to", "to", "subject", "date")


def _strip_html(html_text: str) -> str:
    """Minimal HTML → plain text: unescape entities, remove tags, collapse whitespace."""
    text = html.unescape(html_text)
    text = _TAG_RE.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def _collect_parts(msg: Message) -> list[str]:
    """Recursively collect text content from a (possibly MIME-multipart) message."""
    parts: list[str] = []
    if msg.is_multipart():
        for part in msg.get_payload(decode=False):  # type: ignore[arg-type]
            if isinstance(part, Message):
                parts.extend(_collect_parts(part))
    else:
        ctype = msg.get_content_type()
        if ctype not in ("text/plain", "text/html"):
            return parts
        charset = msg.get_content_charset() or "utf-8"
        raw = msg.get_payload(decode=True)
        if not isinstance(raw, bytes):
            return parts
        try:
            text = raw.decode(charset, errors="replace")
        except (LookupError, UnicodeDecodeError):
            text = raw.decode("utf-8", errors="replace")
        if ctype == "text/html":
            text = _strip_html(text)
        if text.strip():
            parts.append(text.strip())
    return parts


def extract_text(raw_email: str | bytes) -> str:
    """Parse a raw RFC-2822 email and return a single classifier-ready string.

    Layout of the returned string:
        Subject: <value>
        From: <value>
        <blank line>
        <body text>

    The layout is intentional: the byte model sees the same structure for every
    email, making subject-line spam signals easy to learn.
    """
    if isinstance(raw_email, str):
        raw_bytes = raw_email.encode("utf-8", errors="replace")
    else:
        raw_bytes = raw_email

    try:
        msg = email.message_from_bytes(raw_bytes, policy=email.policy.compat32)
    except Exception:
        # Fall back to treating the whole thing as plain text
        return raw_email if isinstance(raw_email, str) else raw_email.decode("utf-8", errors="replace")

    # ── Selected headers ──
    header_lines: list[str] = []
    for key in _HEADER_KEEP:
        val = msg.get(key)
        if val:
            # Decode RFC-2047 encoded words (e.g. =?UTF-8?B?...?=)
            try:
                from email.header import decode_header, make_header
                decoded = str(make_header(decode_header(val)))
            except Exception:
                decoded = val
            header_lines.append(f"{key.capitalize()}: {decoded.strip()}")

    # ── Body ──
    body_parts = _collect_parts(msg)
    body = "\n".join(body_parts) if body_parts else ""

    return "\n".join(header_lines) + "\n\n" + body


def text_to_token_ids(text: str, max_bytes: int) -> list[int]:
    """Convert a string to a list of byte values [0, 255], truncated to *max_bytes*.

    Byte-level encoding needs no external tokeniser and handles any language,
    character encoding, and URL/HTML fragment naturally.
    """
    encoded = text.encode("utf-8", errors="replace")
    return list(encoded[:max_bytes])
