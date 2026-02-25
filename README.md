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

## Model size & benchmark results (M1 Mac)

Default config (home-server optimised):

| Hyperparameter | Value | Rationale |
|---|---|---|
| `n_embd` | 128 | Tunable via Optuna; 128 is the default starting point |
| `n_layer` | 2 | Per-layer params; diminishing returns beyond 2 (logbook Entry 5) |
| `n_head` | 4 | 2 diff-attn groups |
| `mlp_internal_dim_multiplier` | 32 | N=1024 per head — sweet spot (logbook Entry 13) |
| `chunk_size` | 256 | Subject + body opening; 4× faster than T=512 (measured) |
| `max_position` | 4096 | RoPE table; handles emails up to 4 096 bytes |
| `diff_attn` | True | Δ=-0.019 val loss (logbook Entry 15) |
| `attn_window` | 64 | Δ=-0.032 val loss; regularisation (logbook Entry 16) |

Total: **~3.2M parameters**, 12 MB fp32.

### Measured throughput (M1 Mac, `scripts/benchmark.py`)

| Device | B | T | ms/step | Inference | Emails/s |
|---|---|---|---|---|---|
| CPU | 8 | 128 | 193 ms | 8.9 ms | 41/s |
| CPU | 8 | 256 | 432 ms | 20.6 ms | 19/s |
| CPU | 32 | 256 | 1 508 ms | 14.0 ms | 21/s |
| **MPS** | **16** | **128** | **160 ms** | **3.9 ms** | **100/s** |
| **MPS** | **32** | **256** | **643 ms** | **8.4 ms** | **50/s** |

MPS is **2.3× faster** than CPU. Use `--device mps` on Apple Silicon.

### Recommended training runs

```bash
# Quick validation (13 min, MPS B=16 T=128)
uv run python scripts/train.py --device mps --batch 16 --steps 5000

# Quality run (54 min, MPS B=32 T=256)
uv run python scripts/train.py --device mps --batch 32 --steps 5000

# After Optuna tuning (use best_hparams.json params)
uv run python scripts/train.py --device mps --steps 5000 \
    --lr <best_lr> --embd <best_embd> --dropout <best_dropout>
```

### Hyperparameter tuning (Optuna)

Three parameters are worth tuning; everything else is fixed by the parent repo's ablations.

| Parameter | Search range | Why it matters |
|---|---|---|
| `learning_rate` | log-uniform [1e-4, 1e-3] | Biggest impact on convergence |
| `n_embd` | {64, 128, 256} | Capacity vs. speed |
| `dropout` | uniform [0.0, 0.25] | Regularisation for imbalanced corpus |

Tuning runs at T=128 (fast); takes **~20 min for 25 trials** on MPS:

```bash
uv run python scripts/tune.py --device mps --trials 25 --trial-steps 300
```
