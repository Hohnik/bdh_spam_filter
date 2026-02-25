
# Download the training data
download:
  uv run scripts/download_data.py

# Tune the hyperparameters
tune:
  uv run scripts/tune.py --device mps --trials 25 --trial-steps 300

# Train full model
train lr embd dropout: 
  uv run python scripts/train.py --device mps --batch 32 --steps 5000 \
         --lr {{lr}} --embd {{embd}} --dropout {{dropout}}


