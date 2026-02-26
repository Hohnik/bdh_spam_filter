"""BDH Spam Filter — Kaggle training kernel.

This file is a TEMPLATE. kaggle_push.py fills in the __CONF_*__ placeholders
and uploads the result as kernel.py.  Do not edit the placeholders directly;
change the arguments passed to kaggle_push.py instead.

Training config injected at push time:
  n_embd   = __CONF_N_EMBD__
  mlp_mult = __CONF_MLP_MULT__
  dropout  = __CONF_DROPOUT__
  lr       = __CONF_LR__
  steps    = __CONF_STEPS__
  batch    = __CONF_BATCH__
  use_tpu  = __CONF_USE_TPU__

Source dataset slug (on Kaggle):  __CONF_DATASET_SLUG__
"""

# ─── Kernel bootstrap ─────────────────────────────────────────────────────────
import os
import subprocess
import sys
from pathlib import Path

# Install packages that aren't pre-installed on Kaggle GPU/TPU images.
# torch + torchvision are pre-installed; we only need these extras.
_DEPS = ["datasets", "huggingface_hub", "tqdm", "filelock"]
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q"] + _DEPS,
    check=True,
)

# Mount source code from the companion Kaggle dataset.
# kaggle_push.py uploads src/ as a dataset; it is mounted at /kaggle/input/<slug>.
_SRC_DIR = "/kaggle/input/__CONF_DATASET_SLUG__"
if not Path(_SRC_DIR).exists():
    raise RuntimeError(
        f"Source dataset not found at {_SRC_DIR}.\n"
        "Make sure the dataset is added as an input to this kernel."
    )
sys.path.insert(0, _SRC_DIR)

# ─── Device setup ─────────────────────────────────────────────────────────────
_USE_TPU = __CONF_USE_TPU__   # bool injected by kaggle_push.py

if _USE_TPU:
    # torch_xla is pre-installed on Kaggle TPU instances
    try:
        import torch_xla                          # type: ignore[import]
        import torch_xla.core.xla_model as xm    # type: ignore[import]
        _device_str = "xla"
        print(f"Device: TPU  —  {xm.xla_device()}")
    except ImportError:
        print("WARNING: torch_xla not available; falling back to CPU.")
        _device_str = "cpu"
        _USE_TPU    = False
else:
    import torch
    if torch.cuda.is_available():
        _device_str = "cuda"
        print(f"Device: GPU  —  {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        _device_str = "cpu"
        print("Device: CPU (no GPU/TPU found)")

# ─── Training config ──────────────────────────────────────────────────────────
# All values below are filled in by kaggle_push.py at push time.
_N_EMBD   = __CONF_N_EMBD__
_MLP_MULT = __CONF_MLP_MULT__
_DROPOUT  = __CONF_DROPOUT__
_LR       = __CONF_LR__
_STEPS    = __CONF_STEPS__
_BATCH    = __CONF_BATCH__
_WARMUP   = max(200, _STEPS // 10)
_EVAL_EVERY = max(100, _STEPS // 25)

print(
    f"\nConfig:  n_embd={_N_EMBD}  mlp_mult={_MLP_MULT}  dropout={_DROPOUT}  "
    f"lr={_LR}  steps={_STEPS}  batch={_BATCH}"
)
print(f"  MLP internal dim N = {_N_EMBD * _MLP_MULT // 4} per head")

# ─── Data ─────────────────────────────────────────────────────────────────────
from src.data.download import download_all        # noqa: E402
from src.data.dataset import build_dataloaders   # noqa: E402

_DATA_DIR  = Path("/kaggle/working/data")
_CKPT_DIR  = Path("/kaggle/working/checkpoints")
_CKPT_DIR.mkdir(parents=True, exist_ok=True)

print("\nDownloading training data …")
samples = download_all(_DATA_DIR)

from src.model.classifier import BDHSpamClassifier, SpamClassifierConfig  # noqa: E402
from src.training.trainer import Trainer, TrainerConfig                    # noqa: E402

_model_cfg = SpamClassifierConfig(
    n_embd                      = _N_EMBD,
    n_layer                     = 2,
    n_head                      = 4,
    mlp_internal_dim_multiplier = _MLP_MULT,
    dropout                     = _DROPOUT,
    chunk_size                  = 256,
    max_position                = 4096,
    max_email_bytes             = 4096,
)

_train_loader, _val_loader = build_dataloaders(
    samples,
    seq_len         = _model_cfg.chunk_size,
    max_email_bytes = _model_cfg.max_email_bytes,
    batch_size      = _BATCH,
)

# ─── Model ────────────────────────────────────────────────────────────────────
_model = BDHSpamClassifier(_model_cfg)
print(f"\nModel: {_model.parameter_count():,} parameters  on {_device_str.upper()}")

# ─── Train ────────────────────────────────────────────────────────────────────
_trainer_cfg = TrainerConfig(
    learning_rate  = _LR,
    max_steps      = _STEPS,
    warmup_steps   = _WARMUP,
    eval_every     = _EVAL_EVERY,
    checkpoint_dir = str(_CKPT_DIR),
    device         = _device_str,
    use_xla        = _USE_TPU,
)

_trainer = Trainer(_model, _train_loader, _val_loader, _trainer_cfg)
_trainer.train()

# ─── Summary ──────────────────────────────────────────────────────────────────
_checkpoints = sorted(_CKPT_DIR.glob("*.pt"))
if _checkpoints:
    print(f"\nCheckpoints saved ({len(_checkpoints)} file(s)):")
    for p in _checkpoints:
        print(f"  {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")
    print("\nDownload via:  uv run python scripts/kaggle_push.py --download-only")
else:
    print("\nWARNING: No checkpoints found. Training may have failed.")
