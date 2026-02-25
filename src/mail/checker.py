"""MailChecker — orchestrates IMAP client + BDH spam filter.

Workflow for each unseen email:
  1. Fetch raw RFC-2822 bytes from INBOX
  2. Classify with SpamFilter.classify()  → updates live synaptic state ρ (Role B)
  3. If spam AND confidence ≥ auto_learn_threshold:
       a. Move to spam folder
       b. Call SpamFilter.learn(is_spam=True)  → gradient weight update
  4. Log outcome at appropriate level

Auto-learning is gated by confidence to avoid reinforcing uncertain predictions.
Only classifications with confidence > auto_learn_threshold (default 0.85) trigger
a weight update.  This prevents the feedback loop where borderline spam that happens
to be misclassified gets reinforced.

dry_run=True logs everything but makes no IMAP changes and no weight updates.
"""

from __future__ import annotations

import dataclasses
import logging
from email.message import Message

from .imap_client import IMAPClient
from ..filter import SpamFilter, SpamResult

log = logging.getLogger(__name__)


@dataclasses.dataclass
class CheckResult:
    total:     int = 0
    spam:      int = 0
    ham:       int = 0
    moved:     int = 0
    learned:   int = 0   # weight updates applied
    errors:    int = 0

    def __str__(self) -> str:
        return (
            f"{self.total} checked | "
            f"{self.spam} spam (moved: {self.moved}, learned: {self.learned}) | "
            f"{self.ham} ham | "
            f"{self.errors} errors"
        )


def _header(msg: Message, key: str, default: str = "") -> str:
    """Return a decoded header value or *default* if absent."""
    raw = msg.get(key, default)
    if not raw:
        return default
    try:
        import email.header
        parts   = email.header.decode_header(raw)
        decoded = email.header.make_header(parts)
        return str(decoded).strip()
    except Exception:
        return str(raw).strip()


class MailChecker:
    """Fetches unseen emails, classifies them, and moves spam.

    Args:
        spam_filter:          Loaded SpamFilter instance.
        imap_client:          Connected IMAPClient (used as context manager).
        dry_run:              If True, classify and log but do not move or learn.
        auto_learn_threshold: Confidence threshold above which a classification
                              triggers a weight update (default 0.85).
                              Set to 1.1 to disable auto-learning entirely.
    """

    def __init__(
        self,
        spam_filter:          SpamFilter,
        imap_client:          IMAPClient,
        dry_run:              bool  = False,
        auto_learn_threshold: float = 0.85,
    ) -> None:
        self.filter    = spam_filter
        self.client    = imap_client
        self.dry_run   = dry_run
        self.threshold = auto_learn_threshold

    def run(self) -> CheckResult:
        """Process all unseen messages in INBOX. Returns a CheckResult summary."""
        result       = CheckResult()
        spam_folder  = self.client.ensure_spam_folder()
        dry_tag      = " [DRY RUN]" if self.dry_run else ""

        for uid, msg, raw_bytes in self.client.fetch_unseen():
            result.total += 1
            uid_str  = uid.decode()
            subject  = _header(msg, "Subject", "(no subject)")
            sender   = _header(msg, "From",    "(unknown)")
            date     = _header(msg, "Date",    "")

            try:
                classification: SpamResult = self.filter.classify(raw_bytes)
            except Exception as exc:
                result.errors += 1
                log.error("UID %s  classify() failed: %s", uid_str, exc, exc_info=True)
                continue

            if classification.is_spam:
                result.spam += 1
                log.warning(
                    "SPAM%s  conf=%.3f  uid=%s  from=%r  subject=%r  date=%s",
                    dry_tag, classification.confidence, uid_str, sender, subject, date,
                )
                if not self.dry_run:
                    try:
                        self.client.move_to_spam(uid, spam_folder)
                        result.moved += 1
                    except Exception as exc:
                        result.errors += 1
                        log.error("UID %s  move_to_spam failed: %s", uid_str, exc)

                    # Weight update only for high-confidence spam
                    if classification.confidence >= self.threshold:
                        try:
                            self.filter.learn(raw_bytes, is_spam=True)
                            result.learned += 1
                            log.debug("UID %s  model updated (spam)", uid_str)
                        except Exception as exc:
                            log.warning("UID %s  learn() failed: %s", uid_str, exc)
            else:
                result.ham += 1
                log.info(
                    "HAM%s   conf=%.3f  uid=%s  from=%r  subject=%r",
                    dry_tag, classification.confidence, uid_str, sender, subject,
                )

        log.info("Run complete%s: %s", dry_tag, result)
        return result
