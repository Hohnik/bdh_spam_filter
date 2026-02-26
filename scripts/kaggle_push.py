#!/usr/bin/env python3
"""Push BDH spam filter training to Kaggle free GPU / TPU, monitor, download.

Workflow
────────
  1. Read username from ~/.kaggle/kaggle.json (already present)
  2. Upload project src/ as a private Kaggle dataset (create or new version)
  3. Generate kernel.py from the template with your training config embedded
  4. Push the kernel to Kaggle (GPU by default, TPU with --tpu)
  5. Poll status every 30 s with a spinner
  6. Download checkpoints to checkpoints/ when done

Usage
─────
  # First run — uses Optuna best params as defaults:
  uv run python scripts/kaggle_push.py --lr 4.09e-4 --embd 256 --mlp-mult 16 --dropout 0.114

  # TPU (experimental — slower to start but free 30 h/week separately):
  uv run python scripts/kaggle_push.py --tpu

  # Code didn't change, just re-run training:
  uv run python scripts/kaggle_push.py --skip-upload

  # Fetch results from a previous run without re-training:
  uv run python scripts/kaggle_push.py --download-only

Kaggle free quota
─────────────────
  GPU (T4 / P100):  30 h / week   — ~15 min for 5 000 steps  ← recommended
  TPU (v3-8):       30 h / week   — experimental with PyTorch

Requirements
────────────
  ~/.kaggle/kaggle.json must exist (already present for you).
  Internet access must be enabled: kaggle.com/settings → Phone verification.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

_DATASET_SLUG = "bdh-spam-filter-src"
_KERNEL_SLUG  = "bdh-spam-training"


# ─── Auth ─────────────────────────────────────────────────────────────────────

def _get_api():
    """Return an authenticated KaggleApi instance (kaggle 2.0)."""
    try:
        from kaggle import KaggleApi
    except ImportError:
        print("ERROR: kaggle package not installed. Run:  uv add kaggle", file=sys.stderr)
        sys.exit(1)
    api = KaggleApi()
    api.authenticate()
    return api


def _get_username(api) -> str:
    """Read the Kaggle username from the authenticated session."""
    username = api.get_config_value("username")
    if not username:
        print(
            "ERROR: Could not resolve Kaggle username.\n"
            "  Make sure ~/.kaggle/kaggle.json is valid and contains 'username'.",
            file=sys.stderr,
        )
        sys.exit(1)
    return username


# ─── Dataset upload ───────────────────────────────────────────────────────────

def _upload_source_dataset(api, username: str) -> None:
    """Sync src/ to a private Kaggle dataset. Creates it on first run."""
    project_root = Path(__file__).parent.parent
    full_ref     = f"{username}/{_DATASET_SLUG}"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Copy src/ (strip __pycache__ to keep the upload small)
        dst = tmp_path / "src"
        shutil.copytree(project_root / "src", dst)
        for cache in dst.rglob("__pycache__"):
            shutil.rmtree(cache)

        # Metadata (exact format from api.dataset_initialize())
        meta = {
            "title":    "BDH Spam Filter Source",
            "id":       full_ref,
            "licenses": [{"name": "CC0-1.0"}],
        }
        (tmp_path / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

        # Check whether the dataset already exists
        try:
            api.dataset_status(full_ref)
            exists = True
        except Exception:
            exists = False

        if exists:
            print(f"Uploading new dataset version: {full_ref} …")
            api.dataset_create_version(
                str(tmp_path),
                version_notes = "Code update",
                quiet         = False,
                convert_to_csv = False,
                dir_mode      = "zip",
            )
        else:
            print(f"Creating new dataset: {full_ref} …")
            api.dataset_create_new(
                str(tmp_path),
                public        = False,
                quiet         = False,
                convert_to_csv = False,
                dir_mode      = "zip",
            )

    print(f"  Dataset ready → https://www.kaggle.com/datasets/{full_ref}\n")


# ─── Kernel push ──────────────────────────────────────────────────────────────

def _push_kernel(api, username: str, args: argparse.Namespace) -> None:
    """Generate kernel.py from the template and push to Kaggle."""
    template_path = Path(__file__).parent.parent / "kaggle_kernel" / "kernel_template.py"
    template      = template_path.read_text(encoding="utf-8")

    kernel_src = (
        template
        .replace("__CONF_N_EMBD__",       str(args.embd))
        .replace("__CONF_MLP_MULT__",     str(args.mlp_mult))
        .replace("__CONF_DROPOUT__",      str(args.dropout))
        .replace("__CONF_LR__",           str(args.lr))
        .replace("__CONF_STEPS__",        str(args.steps))
        .replace("__CONF_BATCH__",        str(args.batch))
        .replace("__CONF_USE_TPU__",      "True" if args.tpu else "False")
        .replace("__CONF_DATASET_SLUG__", _DATASET_SLUG)
    )

    # kernel-metadata.json — exact field names and string booleans as Kaggle expects
    meta = {
        "id":           f"{username}/{_KERNEL_SLUG}",
        "title":        "BDH Spam Filter Training",
        "code_file":    "kernel.py",
        "language":     "python",
        "kernel_type":  "script",
        "is_private":   "true",
        "enable_gpu":   "false" if args.tpu else "true",
        "enable_tpu":   "true"  if args.tpu else "false",
        "enable_internet": "true",
        "dataset_sources":    [f"{username}/{_DATASET_SLUG}"],
        "competition_sources": [],
        "kernel_sources":     [],
        "model_sources":      [],
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "kernel.py").write_text(kernel_src, encoding="utf-8")
        (tmp_path / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))

        accelerator = "TPU v3-8" if args.tpu else "GPU (T4/P100)"
        print(f"Pushing kernel → {username}/{_KERNEL_SLUG}  [{accelerator}] …")
        response = api.kernels_push(str(tmp_path))

    print(f"  Queued ✓")
    print(f"  Live logs → https://www.kaggle.com/code/{username}/{_KERNEL_SLUG}\n")


# ─── Monitor + download ───────────────────────────────────────────────────────

def _monitor(api, username: str) -> str:
    """Poll kernel status every 30 s. Returns final status string."""
    kernel_ref     = f"{username}/{_KERNEL_SLUG}"
    POLL_INTERVAL  = 30
    TERMINAL       = {"complete", "error", "cancelAcknowledged", "cancelled"}
    spinner        = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    tick           = 0
    elapsed_sec    = 0

    print("Monitoring (Ctrl-C to detach — kernel keeps running on Kaggle) …\n")
    try:
        while True:
            try:
                obj    = api.kernels_status(kernel_ref)
                status = obj.status or "queued"
                mins   = elapsed_sec // 60
                secs   = elapsed_sec % 60
                msg    = (obj.failure_message or "").strip()
                suffix = f"  {msg}" if msg else ""
                print(
                    f"\r  {spinner[tick % len(spinner)]}  "
                    f"{status:<22}  {mins:02d}:{secs:02d}{suffix}",
                    end="", flush=True,
                )
                tick += 1
                if status in TERMINAL:
                    print()
                    return status
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(f"\n  Warning: status check failed ({exc}). Retrying …")

            time.sleep(POLL_INTERVAL)
            elapsed_sec += POLL_INTERVAL

    except KeyboardInterrupt:
        print(
            "\n\nDetached — kernel keeps running on Kaggle.\n"
            f"Download later:  uv run python scripts/kaggle_push.py --download-only"
        )
        sys.exit(0)


def _download(api, username: str, output_dir: Path) -> None:
    """Download kernel output files to output_dir."""
    kernel_ref = f"{username}/{_KERNEL_SLUG}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading output → {output_dir} …")

    try:
        files, _ = api.kernels_output(kernel_ref, str(output_dir), force=True)
    except Exception as exc:
        print(f"  Download failed: {exc}")
        print(f"  Manual:  kaggle kernels output {kernel_ref} -p {output_dir}")
        return

    pts = sorted(output_dir.rglob("*.pt"))
    if not pts:
        print("  No .pt checkpoints in output. Check kernel logs.")
        return

    print(f"  {len(pts)} checkpoint(s) downloaded:")
    for p in pts:
        print(f"    {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")

    # Copy best (non-interrupted) checkpoint to checkpoints/best.pt
    best_candidates = [p for p in pts if "interrupted" not in p.name]
    best = sorted(best_candidates or pts)[-1]
    dest = Path("checkpoints") / "best.pt"
    dest.parent.mkdir(exist_ok=True)
    shutil.copy(best, dest)
    print(f"\n  Best checkpoint → checkpoints/best.pt")
    print("  Ready:  uv run python scripts/check_mail.py")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Push BDH training to Kaggle GPU/TPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--steps",        type=int,   default=5000,  help="Training steps (default: 5000)")
    parser.add_argument("--lr",           type=float, default=3e-4,  help="Learning rate (default: 3e-4)")
    parser.add_argument("--embd",         type=int,   default=128,   help="n_embd model width (default: 128)")
    parser.add_argument("--mlp-mult",     type=int,   default=None,  help="MLP multiplier (auto if omitted)")
    parser.add_argument("--dropout",      type=float, default=0.1,   help="Dropout (default: 0.1)")
    parser.add_argument("--batch",        type=int,   default=32,    help="Batch size (default: 32)")
    parser.add_argument("--tpu",          action="store_true",       help="Use TPU v3-8 instead of GPU")
    parser.add_argument("--skip-upload",  action="store_true",       help="Skip dataset sync (code unchanged)")
    parser.add_argument("--download-only",action="store_true",       help="Download results from last run")
    parser.add_argument("--output-dir",   default="checkpoints",     help="Local dir for downloaded files")
    args = parser.parse_args()

    # Auto-scale mlp_mult so N ≈ 1024 per head regardless of n_embd
    if args.mlp_mult is None:
        args.mlp_mult = max(8, 1024 * 4 // args.embd)

    output_dir = Path(args.output_dir)

    api      = _get_api()
    username = _get_username(api)
    print(f"Logged in as: {username}\n")

    if args.download_only:
        _download(api, username, output_dir)
        return

    if not args.skip_upload:
        _upload_source_dataset(api, username)
    else:
        print(f"Skipping dataset upload (--skip-upload).\n")

    print(
        f"Config:  n_embd={args.embd}  mlp_mult={args.mlp_mult}  "
        f"N={args.embd * args.mlp_mult // 4}/head\n"
        f"         lr={args.lr}  dropout={args.dropout}  "
        f"steps={args.steps}  batch={args.batch}  "
        f"device={'TPU' if args.tpu else 'GPU'}\n"
    )

    _push_kernel(api, username, args)

    final_status = _monitor(api, username)

    if final_status == "complete":
        print("Kernel complete ✓\n")
        _download(api, username, output_dir)
    else:
        err = ""
        try:
            err = api.kernels_status(f"{username}/{_KERNEL_SLUG}").failure_message or ""
        except Exception:
            pass
        print(f"Kernel ended with status: {final_status}")
        if err:
            print(f"  Error: {err}")
        print(f"  Logs → https://www.kaggle.com/code/{username}/{_KERNEL_SLUG}")


if __name__ == "__main__":
    main()
