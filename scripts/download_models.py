#!/usr/bin/env python3
"""
Download pre-trained model weights for Multi-Object Tracking.

This script downloads the required model weights for:
- YOLOX: Object detection
- FastReID: Person re-identification
- BoostTrack: Tracking model weights

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --models yolox fastreid
    python scripts/download_models.py --output-dir /path/to/weights
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path
from typing import NamedTuple


class ModelInfo(NamedTuple):
    name: str
    url: str
    filename: str
    sha256: str | None  # Optional checksum for verification
    subdir: str  # Subdirectory under weights/


# Model registry - add new models here
MODELS: dict[str, ModelInfo] = {
    "yolox_x": ModelInfo(
        name="YOLOX-X (MOT17)",
        url="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth",
        filename="yolox_x.pth",
        sha256=None,  # Add checksum if known
        subdir="yolox",
    ),
    "yolox_x_mot20": ModelInfo(
        name="YOLOX-X (MOT20)",
        url="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth",
        filename="yolox_x_mot20.pth",
        sha256=None,
        subdir="yolox",
    ),
    # FastReID models - placeholder URLs, replace with actual model hosting
    "fastreid_mot17": ModelInfo(
        name="FastReID SBS-S50 (MOT17)",
        url="",  # Add actual URL when available
        filename="mot17_sbs_S50.pth",
        sha256=None,
        subdir="fastreid",
    ),
    "fastreid_mot20": ModelInfo(
        name="FastReID SBS-S50 (MOT20)",
        url="",  # Add actual URL when available
        filename="mot20_sbs_S50.pth",
        sha256=None,
        subdir="fastreid",
    ),
}

# Default models to download
DEFAULT_MODELS = ["yolox_x"]


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.resolve()


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download a file with progress indication."""
    if not url:
        print(f"  ⚠️  No URL configured for {desc}")
        return False

    print(f"  📥 Downloading {desc}...")
    print(f"     URL: {url}")
    print(f"     Destination: {dest}")

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Download with progress
        def report_progress(block_num: int, block_size: int, total_size: int) -> None:
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, downloaded * 100 // total_size)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r     Progress: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)

        urllib.request.urlretrieve(url, dest, reporthook=report_progress)
        print()  # Newline after progress
        return True

    except Exception as e:
        print(f"\n  ❌ Failed to download: {e}")
        return False


def verify_checksum(filepath: Path, expected: str | None) -> bool:
    """Verify file checksum if provided."""
    if expected is None:
        print("     ⚠️  No checksum available, skipping verification")
        return True

    actual = compute_sha256(filepath)
    if actual == expected:
        print("     ✅ Checksum verified")
        return True
    else:
        print(f"     ❌ Checksum mismatch!")
        print(f"        Expected: {expected}")
        print(f"        Got:      {actual}")
        return False


def download_model(model_key: str, weights_dir: Path, force: bool = False) -> bool:
    """Download a single model."""
    if model_key not in MODELS:
        print(f"❌ Unknown model: {model_key}")
        print(f"   Available models: {', '.join(MODELS.keys())}")
        return False

    model = MODELS[model_key]
    dest = weights_dir / model.subdir / model.filename

    print(f"\n🔧 {model.name}")

    if dest.exists() and not force:
        print(f"  ✅ Already exists: {dest}")
        return True

    if not download_file(model.url, dest, model.name):
        return False

    if not verify_checksum(dest, model.sha256):
        dest.unlink(missing_ok=True)
        return False

    print(f"  ✅ Successfully downloaded to {dest}")
    return True


def list_models() -> None:
    """List all available models."""
    print("\n📋 Available Models:\n")
    for key, model in MODELS.items():
        status = "🔗" if model.url else "⚠️  (URL not configured)"
        print(f"  {key:20s} - {model.name} {status}")
    print(f"\nDefault models: {', '.join(DEFAULT_MODELS)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download pre-trained model weights for MOT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=f"Models to download (default: {' '.join(DEFAULT_MODELS)})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for weights (default: <project>/src/CustomBoostTrack/weights)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if file exists",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available models",
    )

    args = parser.parse_args()

    if args.list:
        list_models()
        return 0

    # Determine weights directory
    if args.output_dir:
        weights_dir = args.output_dir.resolve()
    else:
        weights_dir = get_project_root() / "src" / "CustomBoostTrack" / "weights"

    print("=" * 60)
    print("  MOT Model Weights Downloader")
    print("=" * 60)
    print(f"\n📁 Weights directory: {weights_dir}")

    # Determine which models to download
    models_to_download = list(MODELS.keys()) if args.all else args.models

    print(f"📦 Models to download: {', '.join(models_to_download)}")

    # Download models
    success_count = 0
    fail_count = 0

    for model_key in models_to_download:
        if download_model(model_key, weights_dir, force=args.force):
            success_count += 1
        else:
            fail_count += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"  Summary: {success_count} succeeded, {fail_count} failed")
    print("=" * 60)

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
