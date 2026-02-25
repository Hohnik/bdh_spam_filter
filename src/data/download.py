"""Fetch and cache publicly available spam benchmark datasets.

Two sources are used:

1. SpamAssassin Public Corpus (Apache Foundation)
   URL: https://spamassassin.apache.org/old/publiccorpus/
   Contents: raw RFC-2822 emails, one file per message.
   ~4,000 spam  +  ~6,800 ham emails.

2. Enron Spam Dataset (SetFit/enron_spam on Hugging Face)
   ~33,000 emails labelled by ion Androutsopoulos et al.
   Downloaded via the `datasets` library.

Both datasets are standard benchmarks for spam-filter research and are freely
usable for non-commercial purposes.

Returned format for both: list of (text: str, label: int) where label=1 → spam.
"""

from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path
from typing import NamedTuple

import requests
from tqdm import tqdm


# ─── Constants ───────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent.parent / "data"

# SpamAssassin tarballs (all publicly available, no authentication needed)
_SA_BASE = "https://spamassassin.apache.org/old/publiccorpus"
_SPAMASSASSIN_ARCHIVES: list[tuple[str, int]] = [
    # (filename, label)  — label 1 = spam, 0 = ham
    ("20021010_easy_ham.tar.bz2",  0),
    ("20021010_spam.tar.bz2",      1),
    ("20030228_easy_ham.tar.bz2",  0),
    ("20030228_hard_ham.tar.bz2",  0),
    ("20030228_spam.tar.bz2",      1),
    ("20050311_spam_2.tar.bz2",    1),
]


class Sample(NamedTuple):
    text:  str
    label: int   # 1 = spam, 0 = ham


# ─── SpamAssassin ─────────────────────────────────────────────────────────────

def _download_file(url: str, dest: Path) -> None:
    """Download *url* to *dest*, showing a progress bar. Skips if already present."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
    ) as bar:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)
            bar.update(len(chunk))


def _load_spamassassin_archive(path: Path, label: int) -> list[Sample]:
    """Extract all email files from a tar.bz2 archive and return Sample list."""
    samples: list[Sample] = []
    with tarfile.open(path, "r:bz2") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            # SpamAssassin stores emails as plain files; skip index/cmds files
            name = os.path.basename(member.name)
            if name in ("cmds",):
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            raw = f.read()
            try:
                text = raw.decode("utf-8", errors="replace")
            except Exception:
                text = raw.decode("latin-1", errors="replace")
            if text.strip():
                samples.append(Sample(text=text, label=label))
    return samples


def fetch_spamassassin(data_dir: Path = DATA_DIR) -> list[Sample]:
    """Download and parse all SpamAssassin corpus archives.

    Archives are cached under *data_dir/spamassassin/*.
    Returns a flat list of Sample(text, label).
    """
    sa_dir = data_dir / "spamassassin"
    sa_dir.mkdir(parents=True, exist_ok=True)

    samples: list[Sample] = []
    print("Fetching SpamAssassin Public Corpus …")
    for filename, label in _SPAMASSASSIN_ARCHIVES:
        dest = sa_dir / filename
        _download_file(f"{_SA_BASE}/{filename}", dest)
        batch = _load_spamassassin_archive(dest, label)
        tag   = "spam" if label == 1 else "ham"
        print(f"  {filename}: {len(batch)} {tag} messages")
        samples.extend(batch)

    spam_n = sum(1 for s in samples if s.label == 1)
    ham_n  = len(samples) - spam_n
    print(f"SpamAssassin total: {spam_n} spam, {ham_n} ham\n")
    return samples


# ─── Enron Spam (Hugging Face datasets) ───────────────────────────────────────

def fetch_enron(data_dir: Path = DATA_DIR) -> list[Sample]:
    """Download Enron spam dataset via Hugging Face `datasets`.

    Dataset: SetFit/enron_spam
    Fields used: subject, text  →  concatenated as "Subject: …\\n\\n<body>"
    Label: label_num  (1 = spam, 0 = ham)

    Falls back to an empty list with a warning if the datasets library is not
    available or the download fails.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("Warning: `datasets` package not installed — skipping Enron dataset.")
        return []

    cache_dir = data_dir / "enron_cache"
    print("Fetching Enron spam dataset (Hugging Face SetFit/enron_spam) …")
    try:
        ds = load_dataset("SetFit/enron_spam", cache_dir=str(cache_dir))
    except Exception as exc:
        print(f"Warning: could not download Enron dataset ({exc}) — skipping.")
        return []

    samples: list[Sample] = []
    for split in ds.values():
        for row in split:
            subject = row.get("subject", "") or ""
            body    = row.get("text", "")    or ""
            text    = f"Subject: {subject}\n\n{body}" if subject else body
            label   = int(row.get("label_num", row.get("label", 0)))
            if text.strip():
                samples.append(Sample(text=text, label=label))

    spam_n = sum(1 for s in samples if s.label == 1)
    ham_n  = len(samples) - spam_n
    print(f"Enron total: {spam_n} spam, {ham_n} ham\n")
    return samples


# ─── Public API ───────────────────────────────────────────────────────────────

def download_all(data_dir: Path = DATA_DIR) -> list[Sample]:
    """Download SpamAssassin + Enron datasets and return combined sample list.

    Suitable as the full training corpus for initial model training.
    Subsequent continuous learning uses SpamFilter.learn() on individual emails.
    """
    sa  = fetch_spamassassin(data_dir)
    enr = fetch_enron(data_dir)
    combined = sa + enr

    spam_total = sum(1 for s in combined if s.label == 1)
    ham_total  = len(combined) - spam_total
    print(
        f"Combined corpus: {len(combined)} emails  "
        f"({spam_total} spam / {ham_total} ham)"
    )
    return combined
