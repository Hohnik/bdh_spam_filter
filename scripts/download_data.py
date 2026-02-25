#!/usr/bin/env python3
"""Standalone script to pre-download all training data.

Run this once before training so the datasets are cached locally:

    uv run python scripts/download_data.py

Cached under:  ./data/spamassassin/   (SpamAssassin archives)
               ./data/enron_cache/    (Hugging Face datasets cache)
"""

import sys
from pathlib import Path

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.download import download_all

if __name__ == "__main__":
    samples = download_all()
    spam_n  = sum(1 for s in samples if s.label == 1)
    ham_n   = len(samples) - spam_n
    print(f"\nAll data ready: {len(samples)} emails ({spam_n} spam / {ham_n} ham)")
