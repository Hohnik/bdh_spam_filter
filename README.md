# BDH Spam Filter

A self-hosted email spam classifier built on the **Baby Dragon Hatchling (BDH-GPU)**
architecture with continuous learning — no cloud API, runs entirely on a home server.

---

## Why BDH?

Spam evolves. Classic ML filters (Naive Bayes, SVMs) and even fine-tuned transformers
go stale within months as spammers adopt new patterns. BDH solves this through two
mechanisms that work in parallel during deployment:

| Mechanism | Where | What it does |
|---|---|---|
| **Gradient updates** (OnlineLearner) | Model weights | Permanent weight-level learning from user corrections. Replay buffer prevents forgetting. |
| **Synaptic state ρ** (Role B) | In-memory tensor | Fast Hebbian session memory. Accumulates context across all emails classified in the current session. Persisted on `save()`. |

Together these mean: the filter improves every time a user marks a false positive or
false negative, and it remembers session-level patterns (e.g. a wave of phishing from
the same sender domain) without any retraining job.

---

## Architecture Decision: BDH-GPU over BDH-Graph

The BDH paper describes two variants. This project uses **BDH-GPU** (the dense matmul
version) — not BDH-Graph (the sparse graph version). Here is why:

| | BDH-GPU ✅ | BDH-Graph ❌ |
|---|---|---|
| Throughput (same params, Mac CPU) | ~19 000 tok/s | ~888 tok/s |
| Validation loss (500 steps) | **1.925** | 2.755 |
| CPU performance | Fast — BLAS dense matmuls | Slow — `sparse.mm` overhead dominates on CPU |
| "GPU" required? | **No** — "GPU" refers to the dense computation *pattern*, not hardware | No, but sparse ops have poor CPU/MPS support |
| Continuous learning | ✅ Recurrent state + forget gate, tested | ✅ Less tested |

BDH-Graph's advantages (explicit power-law topology, edge-level synaptic state,
biological fidelity) only materialise on neuromorphic hardware (Intel Loihi,
SpiNNaker) or extremely large N where dense matmuls become quadratic. On a home
server CPU, BDH-GPU is 21× faster with significantly better quality.

---

## About the two roles of the BDH recurrent state

This is a subtle but important distinction:

**Role A — Within-email chunking (computational necessity)**
BDH attention is O(T²). Emails longer than `max_seq_len` must be split into chunks.
ρ carries context across chunks so no information is discarded. A **fresh zero state**
is used for every single classification call.

**Role B — Cross-session continuous learning (the key property)**
ρ accumulates Hebbian updates across all emails processed in the current session.
The state is **persistent** — it is owned by `SpamFilter`, saved with `save()`, and
restored with `load()`. This is what prevents drift over time.

---

## Training Data

Two publicly available benchmark corpora are used:

| Dataset | Emails | Source |
|---|---|---|
| **SpamAssassin Public Corpus** | ~4 000 spam + ~6 800 ham | [spamassassin.apache.org](https://spamassassin.apache.org/old/publiccorpus/) |
| **Enron Spam Dataset** | ~33 000 (labelled) | [SetFit/enron_spam on HuggingFace](https://huggingface.co/datasets/SetFit/enron_spam) |

Both are standard benchmarks in spam-filter research and are freely usable for
non-commercial purposes. Download is automated in `scripts/download_data.py`.

---

## Quick start

```bash
# Install dependencies
uv sync

# Download training data (cached locally, ~150 MB)
uv run python scripts/download_data.py

# Train (CPU, ~5-10 min for 3000 steps)
uv run python scripts/train.py --steps 3000 --device cpu

# Run tests
uv run pytest tests/ -v
```

---

## Usage in code

```python
from src.filter import SpamFilter

# Load a trained filter
sf = SpamFilter.load("checkpoints/best.pt")

# Classify an email (raw RFC-2822 string or bytes)
result = sf.classify(raw_email)
print(result.is_spam, result.confidence)   # e.g. True, 0.97

# Correct a mistake → triggers a gradient update + replay
sf.learn(raw_email, is_spam=True)

# Persist updated weights AND synaptic state
sf.save("checkpoints/updated.pt")
```

---

## Project structure

```
src/
  model/
    bdh.py           # BDH-GPU feature encoder (ported from baby_dragon_hatchling)
    classifier.py    # BDHSpamClassifier — encoder + binary head
  data/
    download.py      # Fetch SpamAssassin + Enron corpora
    email_parser.py  # RFC-2822 → clean text → byte token IDs
    dataset.py       # SpamDataset + DataLoader factory
  training/
    trainer.py       # Offline training loop (F1-optimised, cosine LR)
    online.py        # OnlineLearner — gradient updates with replay buffer
  filter.py          # SpamFilter — public interface tying everything together
scripts/
  download_data.py   # Pre-download datasets
  train.py           # CLI training entry point
tests/
  test_model.py      # BDH encoder + classifier unit tests
  test_data.py       # Email parser + dataset unit tests
  test_filter.py     # SpamFilter integration tests
```

---

## Model size

Default config (home-server optimised):

| Hyperparameter | Value | Rationale |
|---|---|---|
| `n_embd` | 128 | 4× smaller than the LM repo; enough for binary classification |
| `n_layer` | 2 | Per-layer params; diminishing returns beyond 2 (see logbook) |
| `n_head` | 4 | 2 diff-attn groups |
| `mlp_internal_dim_multiplier` | 32 | N=1024 — sweet spot quality/speed (logbook Entry 13) |
| `max_seq_len` | 512 | Covers 95%+ of emails in one pass |
| `diff_attn` | True | Δ=-0.019 val loss (logbook Entry 15) |
| `attn_window` | 64 | Δ=-0.032 val loss; acts as regularisation (logbook Entry 16) |

Total: **~450K parameters** — trains in minutes on CPU, inference in milliseconds.
