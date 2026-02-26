#!/usr/bin/env python3
"""Push BDH spam filter training to Kaggle free GPU / TPU, monitor, download.

Workflow
────────
  1. Read KAGGLE_API_TOKEN from .env → write ~/.kaggle/kaggle.json
  2. Authenticate; fetch your Kaggle username from the API
  3. Upload project src/ as a private Kaggle dataset (create or new version)
  4. Generate kernel.py from the template with your training config embedded
  5. Push the kernel to Kaggle (GPU by default, TPU with --tpu)
  6. Poll status every 30 s with a progress bar
  7. Download checkpoint(s) to checkpoints/ when done

Usage
─────
  # First training run on GPU (T4, ~15 min for 5000 steps):
  uv run python scripts/kaggle_push.py

  # With Optuna best params:
  uv run python scripts/kaggle_push.py --lr 4.09e-4 --embd 256 --mlp-mult 16 --dropout 0.114

  # On TPU (v3-8, experimental):
  uv run python scripts/kaggle_push.py --tpu

  # Just download results from a previous run:
  uv run python scripts/kaggle_push.py --download-only

Notes
─────
  Kaggle GPU quota: 30 h/week (T4 or P100)
  Kaggle TPU quota: 30 h/week (TPU v3-8)

  The kernel needs internet access to download SpamAssassin + Enron (~150 MB).
  This requires accepting Kaggle's phone-verification at kaggle.com/settings.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ─── .env loader ──────────────────────────────────────────────────────────────

def _load_env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


# ─── Kaggle auth ──────────────────────────────────────────────────────────────

def _setup_kaggle_auth(env: dict[str, str]) -> str:
    """Write ~/.kaggle/kaggle.json and return the Kaggle username.

    Supports the new single-token format (KGAT_*) as well as the classic
    username + key format.
    """
    token = (
        env.get("KAGGLE_API_TOKEN")
        or os.environ.get("KAGGLE_API_TOKEN")
    )
    username = env.get("KAGGLE_USERNAME") or os.environ.get("KAGGLE_USERNAME")
    key      = env.get("KAGGLE_KEY")      or os.environ.get("KAGGLE_KEY")

    if not token and not (username and key):
        print(
            "ERROR: No Kaggle credentials found.\n"
            "  Add one of these to your .env file:\n"
            "    KAGGLE_API_TOKEN=KGAT_...          (new single-token format)\n"
            "  or\n"
            "    KAGGLE_USERNAME=yourname\n"
            "    KAGGLE_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
            "\n"
            "  Get your token at: https://www.kaggle.com/settings → API",
            file=sys.stderr,
        )
        sys.exit(1)

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(mode=0o700, exist_ok=True)
    kaggle_json = kaggle_dir / "kaggle.json"

    if token:
        # New format: single KGAT_* token
        creds = {"token": token}
    else:
        creds = {"username": username, "key": key}

    kaggle_json.write_text(json.dumps(creds))
    kaggle_json.chmod(0o600)

    # Authenticate and resolve username
    try:
        from kaggle.api.kaggle_api_extended import KaggleApiExtended  # type: ignore
        api = KaggleApiExtended()
        api.authenticate()
        resolved = api.get_config_value("username")
        if resolved:
            return resolved
    except Exception as exc:
        print(f"WARNING: Could not resolve Kaggle username automatically: {exc}")

    if username:
        return username

    print(
        "ERROR: Could not determine your Kaggle username.\n"
        "  Add KAGGLE_USERNAME=yourname to your .env file.",
        file=sys.stderr,
    )
    sys.exit(1)


# ─── Dataset upload ───────────────────────────────────────────────────────────

_DATASET_SLUG = "bdh-spam-filter-src"

def _upload_source_dataset(api, username: str) -> str:
    """Upload src/ as a private Kaggle dataset. Returns the dataset slug."""
    from tqdm import tqdm

    full_slug    = f"{username}/{_DATASET_SLUG}"
    project_root = Path(__file__).parent.parent

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Copy src/ tree
        src_dest = tmp_path / "src"
        shutil.copytree(project_root / "src", src_dest)
        # Remove __pycache__ to keep upload small
        for cache in src_dest.rglob("__pycache__"):
            shutil.rmtree(cache)

        # Dataset metadata
        meta = {
            "title":    "BDH Spam Filter Source",
            "id":       full_slug,
            "licenses": [{"name": "other"}],
        }
        (tmp_path / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

        # Try to create; if the slug already exists Kaggle returns HTTP 409/400.
        # On failure, fall back to creating a new version.
        try:
            print(f"Creating dataset {full_slug} …")
            api.dataset_create_new(str(tmp_path), public=False, quiet=False, dir_mode="zip")
        except Exception as create_err:
            err_str = str(create_err).lower()
            if any(k in err_str for k in ("already", "exists", "conflict", "400", "409")):
                print(f"  Dataset exists — uploading new version …")
                api.dataset_create_version(
                    str(tmp_path), version_notes="Code update", quiet=False, dir_mode="zip"
                )
            else:
                raise

    print(f"  Dataset ready: https://www.kaggle.com/datasets/{full_slug}\n")
    return full_slug


# ─── Kernel generation ────────────────────────────────────────────────────────

_KERNEL_SLUG = "bdh-spam-training"

def _build_kernel_dir(
    tmp_path:     Path,
    username:     str,
    dataset_slug: str,
    args:         argparse.Namespace,
) -> None:
    """Write kernel.py and kernel-metadata.json into *tmp_path*."""
    template_path = (
        Path(__file__).parent.parent / "kaggle_kernel" / "kernel_template.py"
    )
    template = template_path.read_text(encoding="utf-8")

    # Fill in config placeholders
    kernel_src = (
        template
        .replace("__CONF_N_EMBD__",        str(args.embd))
        .replace("__CONF_MLP_MULT__",       str(args.mlp_mult))
        .replace("__CONF_DROPOUT__",        str(args.dropout))
        .replace("__CONF_LR__",             str(args.lr))
        .replace("__CONF_STEPS__",          str(args.steps))
        .replace("__CONF_BATCH__",          str(args.batch))
        .replace("__CONF_USE_TPU__",        "True" if args.tpu else "False")
        .replace("__CONF_DATASET_SLUG__",   _DATASET_SLUG)
    )
    (tmp_path / "kernel.py").write_text(kernel_src, encoding="utf-8")

    # Kernel metadata
    meta = {
        "id":           f"{username}/{_KERNEL_SLUG}",
        "title":        "BDH Spam Filter Training",
        "code_file":    "kernel.py",
        "language":     "python",
        "kernel_type":  "script",
        "is_private":   True,
        "enable_gpu":   not args.tpu,
        "enable_tpu":   args.tpu,
        "enable_internet": True,
        "dataset_sources":   [f"{username}/{_DATASET_SLUG}"],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (tmp_path / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))


def _push_kernel(api, username: str, dataset_slug: str, args: argparse.Namespace) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _build_kernel_dir(tmp_path, username, dataset_slug, args)
        print(f"Pushing kernel {username}/{_KERNEL_SLUG} …")
        api.kernels_push(str(tmp_path))
    accelerator = "TPU v3-8" if args.tpu else "GPU T4"
    print(f"  Kernel queued on {accelerator}.")
    print(f"  Live logs: https://www.kaggle.com/code/{username}/{_KERNEL_SLUG}\n")


# ─── Monitor + download ───────────────────────────────────────────────────────

def _monitor_and_download(api, username: str, output_dir: Path) -> None:
    """Poll kernel status and download checkpoints when done."""
    from tqdm import tqdm

    POLL_INTERVAL = 30   # seconds

    print("Monitoring kernel … (Ctrl-C to detach, kernel keeps running)\n")
    terminal_states = {"complete", "error", "cancelAcknowledged", "cancelled"}
    spinner         = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    i = 0

    try:
        while True:
            try:
                status_obj = api.kernel_status(username, _KERNEL_SLUG)
                status     = status_obj.status
                run_time   = getattr(status_obj, "totalRunningTimeSeconds", 0) or 0
                mins       = int(run_time) // 60
                secs       = int(run_time) % 60
                print(
                    f"\r  {spinner[i % len(spinner)]}  status={status:<20} "
                    f"runtime={mins:02d}:{secs:02d}",
                    end="", flush=True,
                )
                i += 1

                if status in terminal_states:
                    print()
                    break
            except KeyboardInterrupt:
                print(
                    "\n\nDetached. Kernel still running on Kaggle.\n"
                    "Download later with:  uv run python scripts/kaggle_push.py --download-only"
                )
                return
            except Exception as exc:
                print(f"\n  Warning: status check failed ({exc}). Retrying …")

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\nDetached.")
        return

    if status == "complete":
        print(f"  ✓ Kernel complete!\n")
        _download_output(api, username, output_dir)
    elif status == "error":
        print(
            f"  ✗ Kernel failed. Check logs at:\n"
            f"    https://www.kaggle.com/code/{username}/{_KERNEL_SLUG}\n"
        )
    else:
        print(f"  Status: {status}")


def _download_output(api, username: str, output_dir: Path) -> None:
    """Download kernel output (checkpoints) to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading output to {output_dir} …")
    try:
        api.kernel_output(username, _KERNEL_SLUG, path=str(output_dir), force=True)
        pts = list(output_dir.rglob("*.pt"))
        if pts:
            print(f"  ✓ {len(pts)} checkpoint(s) downloaded:")
            for p in sorted(pts):
                print(f"      {p.relative_to(output_dir)}  ({p.stat().st_size / 1e6:.1f} MB)")
            # Copy the best checkpoint to checkpoints/best.pt for check_mail.py
            best_candidates = [p for p in pts if "interrupted" not in p.name]
            if best_candidates:
                best = sorted(best_candidates)[-1]
                dest = Path("checkpoints") / "best.pt"
                dest.parent.mkdir(exist_ok=True)
                shutil.copy(best, dest)
                print(f"\n  Copied best checkpoint → checkpoints/best.pt")
                print("  Ready to run:  uv run python scripts/check_mail.py")
        else:
            print("  No .pt files found in output. Check kernel logs.")
    except Exception as exc:
        print(f"  Download failed: {exc}")
        print(
            f"  Manual download:\n"
            f"    kaggle kernels output {username}/{_KERNEL_SLUG} -p checkpoints/"
        )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Push BDH training to Kaggle GPU/TPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--steps",    type=int,   default=5000,       help="Training steps")
    parser.add_argument("--lr",       type=float, default=3e-4,       help="Learning rate")
    parser.add_argument("--embd",     type=int,   default=128,        help="n_embd")
    parser.add_argument("--mlp-mult", type=int,   default=None,       help="MLP multiplier (auto if omitted)")
    parser.add_argument("--dropout",  type=float, default=0.1,        help="Dropout")
    parser.add_argument("--batch",    type=int,   default=32,         help="Batch size")
    parser.add_argument("--tpu",      action="store_true",            help="Use TPU instead of GPU")
    parser.add_argument("--skip-upload", action="store_true",         help="Skip dataset upload (code unchanged)")
    parser.add_argument("--download-only", action="store_true",       help="Only download results from last run")
    parser.add_argument("--output-dir", default="checkpoints",        help="Local dir for downloaded checkpoints")
    parser.add_argument("--env-file", default=".env",                 help="Path to .env file")
    args = parser.parse_args()

    # Auto-scale mlp_mult to keep N ≈ 1024 (same as local train.py)
    if args.mlp_mult is None:
        args.mlp_mult = max(8, 1024 * 4 // args.embd)

    project_root = Path(__file__).parent.parent
    env          = _load_env(project_root / args.env_file)

    # ── Auth ──────────────────────────────────────────────────────────────────
    username = _setup_kaggle_auth(env)
    print(f"Authenticated as: {username}\n")

    try:
        from kaggle.api.kaggle_api_extended import KaggleApiExtended  # type: ignore
        api = KaggleApiExtended()
        api.authenticate()
    except ImportError:
        print("ERROR: kaggle package not installed. Run:  uv add kaggle")
        sys.exit(1)

    output_dir = project_root / args.output_dir

    # ── Download-only mode ────────────────────────────────────────────────────
    if args.download_only:
        _download_output(api, username, output_dir)
        return

    # ── Upload source dataset ─────────────────────────────────────────────────
    if not args.skip_upload:
        dataset_slug = _upload_source_dataset(api, username)
    else:
        dataset_slug = f"{username}/{_DATASET_SLUG}"
        print(f"Skipping dataset upload. Using: {dataset_slug}")

    # ── Push kernel ───────────────────────────────────────────────────────────
    print(
        f"Training config:\n"
        f"  n_embd={args.embd}  mlp_mult={args.mlp_mult}  "
        f"dropout={args.dropout}  lr={args.lr}\n"
        f"  steps={args.steps}  batch={args.batch}  "
        f"device={'TPU' if args.tpu else 'GPU'}\n"
        f"  MLP internal dim N = {args.embd * args.mlp_mult // 4} per head\n"
    )
    _push_kernel(api, username, dataset_slug, args)

    # ── Monitor ───────────────────────────────────────────────────────────────
    _monitor_and_download(api, username, output_dir)


if __name__ == "__main__":
    main()
