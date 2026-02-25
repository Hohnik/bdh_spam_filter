#!/usr/bin/env python3
"""Check inbox for spam, move detected spam to Spam folder, log all actions.

One-shot mode (default — suitable for cron):
    uv run python scripts/check_mail.py

Daemon mode (runs every N minutes, reconnects each cycle):
    uv run python scripts/check_mail.py --daemon --interval 5

Credentials are read from the project .env file (EMAIL= / PASSWORD=) or from
environment variables EMAIL and EMAIL_PASSWORD.  Never pass passwords as CLI args.

T-Online server settings are pre-configured. Override with --host / --port /
--spam-folder if needed.

Logging:
  Console: INFO and above
  File:    logs/spam_filter.log — DEBUG and above, daily rotation, 30 days kept

Exit codes:
  0  success (zero or more spam found and moved)
  1  configuration error (no model, no credentials)
  2  connection error
"""

from __future__ import annotations

import argparse
import imaplib
import logging
import logging.handlers
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.filter import SpamFilter
from src.mail.imap_client import IMAPClient, IMAPConfig
from src.mail.checker import MailChecker


# ─── Logging setup ────────────────────────────────────────────────────────────

def _setup_logging(log_dir: Path, verbose: bool) -> logging.Logger:
    """Configure root logger with a rotating file handler + console handler."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "spam_filter.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # ── File handler: DEBUG+, daily rotation, keep 30 days ──
    fh = logging.handlers.TimedRotatingFileHandler(
        log_path,
        when       = "midnight",
        backupCount = 30,
        encoding   = "utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        fmt     = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(fh)

    # ── Console handler: INFO+ (DEBUG if --verbose) ──
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(logging.Formatter(
        fmt     = "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt = "%H:%M:%S",
    ))
    root.addHandler(ch)

    # Silence noisy third-party loggers
    for lib in ("urllib3", "datasets", "filelock", "huggingface_hub"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    return logging.getLogger("check_mail")


# ─── Credential loading ───────────────────────────────────────────────────────

def _load_env_file(path: Path) -> dict[str, str]:
    """Parse a .env file (KEY=value lines). Ignores comments and blank lines."""
    result: dict[str, str] = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        result[key.strip()] = value.strip().strip('"').strip("'")
    return result


def _load_credentials(project_root: Path) -> tuple[str, str]:
    """Return (email, password) from .env or environment variables.

    Priority:
      1. Environment variables EMAIL and EMAIL_PASSWORD
      2. .env file in project root (keys: EMAIL and PASSWORD)

    Raises SystemExit(1) with a clear message if credentials are missing.
    """
    env = _load_env_file(project_root / ".env")

    email    = os.environ.get("EMAIL")    or env.get("EMAIL")
    password = os.environ.get("EMAIL_PASSWORD") or env.get("PASSWORD")

    if not email or not password:
        print(
            "ERROR: Email credentials not found.\n"
            "  Option A — set environment variables:\n"
            "    export EMAIL=you@t-online.de\n"
            "    export EMAIL_PASSWORD=yourpassword\n"
            "  Option B — create a .env file in the project root:\n"
            "    EMAIL=you@t-online.de\n"
            "    PASSWORD=yourpassword\n",
            file=sys.stderr,
        )
        sys.exit(1)

    return email, password


# ─── Single check cycle ───────────────────────────────────────────────────────

def _run_cycle(
    imap_cfg:  IMAPConfig,
    sf:        SpamFilter,
    dry_run:   bool,
    learn_thr: float,
    log:       logging.Logger,
) -> bool:
    """Connect, check inbox, disconnect. Returns True on success."""
    try:
        with IMAPClient(imap_cfg) as client:
            checker = MailChecker(
                spam_filter          = sf,
                imap_client          = client,
                dry_run              = dry_run,
                auto_learn_threshold = learn_thr,
            )
            checker.run()
        return True
    except imaplib.IMAP4.error as exc:
        log.error("IMAP connection/protocol error: %s", exc)
        return False
    except Exception as exc:
        log.error("Unexpected error in check cycle: %s", exc, exc_info=True)
        return False


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BDH spam filter — check inbox and move spam",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--host",        default=None,
                        help="IMAP host (default: secureimap.t-online.de)")
    parser.add_argument("--port",        type=int, default=993,
                        help="IMAP port (default: 993)")
    parser.add_argument("--spam-folder", default="Spam",
                        help="Target folder for spam (default: Spam)")
    parser.add_argument("--checkpoint",  default=None,
                        help="Path to SpamFilter checkpoint .pt file")
    parser.add_argument("--log-dir",     default="logs",
                        help="Directory for log files (default: logs/)")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Classify and log but do not move emails or update model")
    parser.add_argument("--learn-threshold", type=float, default=0.85,
                        help="Confidence threshold for auto weight-update (default: 0.85)")
    parser.add_argument("--daemon",      action="store_true",
                        help="Run continuously (reconnect each cycle)")
    parser.add_argument("--interval",    type=int, default=5,
                        help="Daemon check interval in minutes (default: 5)")
    parser.add_argument("--verbose",     action="store_true",
                        help="Print DEBUG-level output to console")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    log = _setup_logging(project_root / args.log_dir, args.verbose)

    # ── Credentials ───────────────────────────────────────────────────────────
    email_addr, password = _load_credentials(project_root)
    log.info("Credentials loaded for %s", email_addr)

    # ── IMAP config ───────────────────────────────────────────────────────────
    if args.host:
        imap_cfg = IMAPConfig(
            host         = args.host,
            port         = args.port,
            username     = email_addr,
            password     = password,
            spam_folder  = args.spam_folder,
        )
    else:
        imap_cfg = IMAPConfig.t_online(email_addr, password)
        imap_cfg.spam_folder = args.spam_folder

    # ── SpamFilter ────────────────────────────────────────────────────────────
    ckpt_search = [
        args.checkpoint,
        str(project_root / "checkpoints" / "best.pt"),
    ]
    ckpt_path: str | None = next(
        (p for p in ckpt_search if p and Path(p).exists()), None
    )
    if ckpt_path is None:
        log.error(
            "No trained model found. "
            "Run 'uv run python scripts/train.py' first, "
            "or pass --checkpoint path/to/model.pt"
        )
        sys.exit(1)

    log.info("Loading SpamFilter from %s …", ckpt_path)
    sf = SpamFilter.load(ckpt_path)
    sf.threshold = 0.5
    log.info("SpamFilter ready (threshold=%.2f)", sf.threshold)

    if args.dry_run:
        log.info("DRY RUN mode — no emails will be moved and model will not be updated.")

    # ── Run ───────────────────────────────────────────────────────────────────
    def _save_state() -> None:
        if not args.dry_run:
            sf.save(ckpt_path)
            log.debug("Filter state saved to %s", ckpt_path)

    if args.daemon:
        log.info("Daemon mode: checking every %d minute(s). Ctrl-C to stop.", args.interval)
        try:
            while True:
                log.info("── Check cycle ──")
                _run_cycle(imap_cfg, sf, args.dry_run, args.learn_threshold, log)
                _save_state()
                log.info("Next check in %d minute(s).", args.interval)
                time.sleep(args.interval * 60)
        except KeyboardInterrupt:
            log.info("Daemon stopped by user.")
    else:
        success = _run_cycle(imap_cfg, sf, args.dry_run, args.learn_threshold, log)
        _save_state()
        sys.exit(0 if success else 2)


if __name__ == "__main__":
    main()
