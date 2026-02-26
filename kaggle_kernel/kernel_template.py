"""BDH Spam Filter — Kaggle training kernel.

Self-contained: the entire src/ package is embedded as a base64 gzip tarball
(__CONF_SRC_TAR_B64__).  No Kaggle dataset dependency — no race conditions,
no mounting delays, no version mismatches.

kaggle_push.py fills in all __CONF_*__ placeholders before pushing.
"""

# ─── Bootstrap: extract embedded source ──────────────────────────────────────
import base64
import io
import os
import subprocess
import sys
import tarfile
from pathlib import Path

_SRC_B64 = "__CONF_SRC_TAR_B64__"

_WORK = Path("/kaggle/working")
_WORK.mkdir(exist_ok=True)

print("Extracting embedded source …", flush=True)
_tar_bytes = base64.b64decode(_SRC_B64)
with tarfile.open(fileobj=io.BytesIO(_tar_bytes), mode="r:gz") as _tar:
    _tar.extractall(str(_WORK))

# Verify extraction
_src_pkg = _WORK / "src"
if not _src_pkg.exists():
    raise RuntimeError(
        f"src/ not found after extraction. "
        f"Contents of {_WORK}: {list(_WORK.iterdir())}"
    )
sys.path.insert(0, str(_WORK))
print(f"  Source ready at {_src_pkg} ({len(list(_src_pkg.rglob('*.py')))} files)")

# ─── Install missing packages ─────────────────────────────────────────────────
_DEPS = ["datasets", "huggingface_hub", "tqdm", "filelock"]
print(f"Installing: {' '.join(_DEPS)} …", flush=True)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q"] + _DEPS,
    check=True,
)

# ─── Device setup ─────────────────────────────────────────────────────────────
_USE_TPU = __CONF_USE_TPU__

if _USE_TPU:
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
        print("Device: CPU (no GPU/TPU found — training will be slow)")

# ─── Training config ──────────────────────────────────────────────────────────
_N_EMBD   = __CONF_N_EMBD__
_MLP_MULT = __CONF_MLP_MULT__
_DROPOUT  = __CONF_DROPOUT__
_LR       = __CONF_LR__
_STEPS    = __CONF_STEPS__
_BATCH    = __CONF_BATCH__
_WARMUP   = max(200, _STEPS // 10)
_EVAL_EVERY = max(100, _STEPS // 25)

print(
    f"\nConfig:  n_embd={_N_EMBD}  mlp_mult={_MLP_MULT}  "
    f"N={_N_EMBD * _MLP_MULT // 4}/head\n"
    f"         dropout={_DROPOUT}  lr={_LR}  "
    f"steps={_STEPS}  batch={_BATCH}  device={_device_str.upper()}"
)

# ─── Data ─────────────────────────────────────────────────────────────────────
from src.data.download import download_all        # noqa: E402
from src.data.dataset import build_dataloaders   # noqa: E402

_DATA_DIR = _WORK / "data"
_CKPT_DIR = _WORK / "checkpoints"
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
print(f"\nModel: {_model.parameter_count():,} parameters on {_device_str.upper()}")

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
    print(f"\nCheckpoints ({len(_checkpoints)} file(s)):")
    for p in _checkpoints:
        print(f"  {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")
    print("\nDownload:  uv run python scripts/kaggle_push.py --download-only")
else:
    print("\nWARNING: No checkpoints saved (val F1 may not have improved).")
