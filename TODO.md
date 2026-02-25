# TODO

## Done ✅
- [x] Architecture decision: BDH-GPU over BDH-Graph (with rationale)
- [x] Clarify two distinct roles of the BDH recurrent state (Role A vs Role B)
- [x] BDH-GPU encoder (src/model/bdh.py) — all improvements from baby_dragon_hatchling
- [x] BDHSpamClassifier with correct Role A / Role B state handling
- [x] SpamAssassin + Enron data download pipeline
- [x] RFC-2822 / MIME email parser (stdlib only, no BeautifulSoup)
- [x] SpamDataset + stratified DataLoader factory
- [x] Offline Trainer with cosine LR, F1 metrics, best-F1 checkpointing
- [x] OnlineLearner with replay buffer (balanced spam/ham deques)
- [x] SpamFilter public interface (classify, learn, save, load, reset_state)
- [x] Tests: model, data, filter (all without network I/O)
- [x] README documenting architecture decision and two-role state design
- [x] LOGBOOK entry

## Next Steps
- [x] Run tests and fix any issues: `uv run pytest tests/ -v` → **34/34 passing**
- [x] Install dependencies: `uv sync`
- [ ] Download data: `uv run python scripts/download_data.py`
- [ ] First training run: `uv run python scripts/train.py --steps 500 --device cpu`
- [ ] Evaluate actual F1 / precision / recall on held-out test set
- [ ] Measure inference latency on a single email

## Future Work
- [ ] Milter integration: connect filter to a Postfix/Dovecot server via pymilter
- [ ] CLI tool: `bdh-spam classify < email.eml`
- [ ] Threshold tuning: ROC curve, choose threshold by desired precision/recall trade-off
- [ ] Periodic full retraining on accumulated user-labelled examples (weekly cron job)
- [ ] TREC 2007 evaluation to benchmark against published state-of-the-art
- [ ] Quantisation (int8) for lower memory footprint on constrained home servers
- [ ] Experiment: larger model (n_embd=256) now that we have more training data than the LM experiments
