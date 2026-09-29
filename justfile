[private]
default:
  @just --list --unsorted

# Check inbox for SPAM mail
check:
  uv run scripts/check_mail.py --dry-run

# Download the training data
download:
  uv run scripts/download_data.py

# Tune the hyperparameters
tune:
  uv run scripts/tune.py --device mps --trials 25 --trial-steps 300

# Train full model
train: 
  uv run scripts/kaggle_push.py --lr 4.09e-4 --embd 256 --mlp-mult 16 --dropout 0.114

download-model: 
  uv run scripts/kaggle_push.py --download-only
