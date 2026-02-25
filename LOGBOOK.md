# Logbook

## Entry 1 — Initial Architecture Decision & Project Foundation (2026-02-25)

### Goal
Bootstrap the BDH spam filter project: choose the right BDH variant, design the
architecture, implement all core components, and create the project plan.

### Architecture Decision: BDH-GPU over BDH-Graph

Evaluated both variants from `~/coding/baby_dragon_hatchling/src/`:

**BDH-GPU** (`bdh.py`):
- Dense matmul operations (mean-field / "radio network" interaction)
- State ρ ∈ R^{n_head × N × D} (compressed outer product)
- Benchmarked: ~19 000 tok/s on M1 CPU at 1.6M params

**BDH-Graph** (`bdh_graph.py`):
- Sparse graph propagation via `torch.sparse.mm`
- Power-law connectivity (α=1.5), edge-level synaptic state
- Benchmarked: ~888 tok/s on M1 CPU at 1.6M params (21× slower)
- Val loss 2.755 vs BDH-GPU 1.925 at equal params and 500 steps

**Verdict: BDH-GPU.** On CPU (home server), BLAS dense matmuls dominate sparse ops.
Despite the confusing name, BDH-GPU does NOT require a GPU. The name refers to the
dense computation *pattern* vs. the graph's wire-based communication. BDH-Graph's
advantages only appear on neuromorphic hardware (Loihi, SpiNNaker) at very large N.

### Key Design Insight: Two Roles of the BDH Recurrent State

During implementation discovered a subtle but critical distinction:

**Role A (within-email chunking):** Computational necessity. O(T²) attention means
long emails must be chunked. State ρ makes chunking lossless. Fresh zero state per
classification call.

**Role B (cross-session continuous learning):** The paper's intended feature. State ρ
accumulates Hebbian updates across all emails processed in the current session.
Persistent — owned by SpamFilter, saved/loaded with the model. This is what prevents
drift as spam evolves. Initially implemented incorrectly (fresh state per call),
corrected after reviewing the distinction with the project owner.

### Implementation Completed

**src/model/bdh.py**
  - BDH-GPU encoder ported from baby_dragon_hatchling with all improvements:
    complex RoPE, flat encode matmul, einsum encode_v, diff_attn, local attn window
  - Adapted to return hidden states (B, T, D) instead of LM logits
  - Stateless forward (state=None) + stateful forward (state≠None)
  - init_state() for explicit zero-state initialisation

**src/model/classifier.py**
  - BDHSpamClassifier = BDH encoder + Linear(D, 1) + sigmoid
  - _encode() handles Role A chunking transparently
  - predict() accepts live_state for Role B
  - from_checkpoint() / save_checkpoint() for persistence

**src/data/download.py**
  - SpamAssassin Public Corpus: 6 tar.bz2 archives, ~10 800 emails
  - Enron spam (SetFit/enron_spam via HuggingFace datasets): ~33 000 emails
  - Streaming download with tqdm progress, local caching

**src/data/email_parser.py**
  - RFC-2822 / MIME parser using stdlib `email` module (no extra deps)
  - Strips HTML tags, decodes RFC-2047 encoded headers, drops binary attachments
  - text_to_token_ids(): string → byte values [0,255], no tokeniser needed

**src/data/dataset.py**
  - SpamDataset: fixed-length padding/truncation, PAD_ID=0
  - build_dataloaders(): stratified train/val split, configurable batch size

**src/training/trainer.py**
  - Cosine LR with linear warmup
  - Metrics: F1, precision, recall (not just accuracy — corpus is imbalanced)
  - Checkpoint on best val F1

**src/training/online.py**
  - OnlineLearner: single gradient step per user correction
  - Separate spam/ham deque replay buffers (balanced replay, configurable size)
  - Learning rate 1e-5 (30× smaller than training LR) to prevent forgetting
  - Gradient clipping 0.5

**src/filter.py**
  - SpamFilter: public interface tying everything together
  - classify() → SpamResult(is_spam, confidence, threshold)
  - learn() → gradient update + replay
  - save() / load() persist both weights and synaptic state ρ
  - reset_state() to start a fresh session
  - train_new() all-in-one factory for first-time setup

**tests/**
  - test_model.py: encoder shapes, stateful correctness, gradient flow, checkpoint
  - test_data.py: email parsing edge cases, dataset shapes, byte-range validity
  - test_filter.py: classify shapes, Role B state accumulation, learn stability, save/load

### Bugs found and fixed during test run

**Bug 1 — RoPE table exhausted on second chunk**
`max_seq_len` in `SpamClassifierConfig` was doing double-duty as both the chunk
size (tokens per forward pass) and the RoPE table size (max absolute position).
When processing chunk 2 with `pos_offset = chunk_size`, the RoPE slice
`freq[:, :, 64:128, :]` was empty because the table only went to index 63.

Fix: split into two distinct fields:
  - `chunk_size`    — tokens per forward pass (memory/compute control)
  - `max_position`  — RoPE table size; must be ≥ max_email_bytes

`to_bdh_config()` now passes `max_seq_len = max_position` to the BDH core.
Caught by new test `test_chunked_state_is_propagated`.

**Bug 2 — filter._preprocess() truncated to chunk_size, not max_email_bytes**
Long emails were silently truncated to 512 bytes before being passed to
`_encode()`, defeating the entire chunking mechanism. Fix: pad to chunk_size
only when the email is shorter; pass the full email (up to max_email_bytes)
and let `_encode()` split it into chunks.

**Bug 3 — test_save_and_load_round_trip state timeline**
`filter_instance.save()` was called after `classify(HAM_EMAIL)` so the saved
state included HAM context; on reload the model was in a different state when
classifying HAM again. Fixed by saving before classifying HAM so both original
and loaded start from the same state when processing the reference email.

### Model Config (home-server optimised)

| Parameter | Value | Basis |
|---|---|---|
| n_embd | 128 | 4× smaller than LM repo; sufficient for binary classification |
| n_layer | 2 | Per-layer params; no improvement beyond 2 (logbook Entry 5) |
| n_head | 4 | 2 diff-attn groups |
| mlp_internal_dim_multiplier | 32 | N=1024; sweet spot (logbook Entry 13) |
| max_seq_len | 512 | Covers 95%+ of emails in single pass |
| diff_attn | True | Δ=-0.019 val loss (logbook Entry 15) |
| attn_window | 64 | Δ=-0.032 val loss (logbook Entry 16) |

Estimated total parameters: ~450K. Inference: <1ms on modern CPU.
