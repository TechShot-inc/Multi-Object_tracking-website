#!/usr/bin/env python3
"""scripts/download_models.py

Download pre-trained model weights used by this repository.

Supports both direct HTTP(S) links and Google Drive share links.

Default output directory:
    - <project-root>/models

Examples:
    python scripts/download_models.py --list
    python scripts/download_models.py --models yolo11 yolo12 osnet
    python scripts/download_models.py --output-dir ./models --force
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import urllib.request
import urllib.parse
import http.cookiejar
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
    "osnet": ModelInfo(
        name="OSNet-AIN ReID (osnet_ain_ms_m_c)",
        url="https://drive.google.com/file/d/14lzvwlPPCVr7Ldseu17ga7vo4QMURQ67/view?usp=sharing",
        filename="osnet_ain_ms_m_c.pth.tar",
        sha256=None,
        subdir=".",
    ),
    "yolo11": ModelInfo(
        name="YOLO11 (ensemble detector #1)",
        url="https://drive.google.com/file/d/1Skoy3bODg1EHwUi7IAdXJ9X0D5HDHTyv/view?usp=sharing",
        filename="yolo11x.pt",
        sha256=None,
        subdir=".",
    ),
    "yolo12": ModelInfo(
        name="YOLO12 (ensemble detector #2)",
        url="https://drive.google.com/file/d/1-9gkRRWnWGv7ZI4O3qS9UoarUrGlaPZs/view?usp=sharing",
        filename="yolo12x.pt",
        sha256=None,
        subdir=".",
    ),
}

# Default models to download
DEFAULT_MODELS = ["yolo11", "yolo12", "osnet"]


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


def _extract_gdrive_file_id(url: str) -> str | None:
    """Extract a Google Drive file id from common share URL formats."""
    if "drive.google.com" not in url:
        return None

    # Format: https://drive.google.com/file/d/<id>/view?...
    marker = "/file/d/"
    if marker in url:
        tail = url.split(marker, 1)[1]
        file_id = tail.split("/", 1)[0]
        return file_id or None

    # Format: https://drive.google.com/open?id=<id>
    try:
        parsed = urllib.parse.urlparse(url)
        qs = urllib.parse.parse_qs(parsed.query)
        file_id = (qs.get("id") or [None])[0]
        return file_id
    except Exception:
        return None


def _download_google_drive(url: str, dest: Path, desc: str = "") -> bool:
    """Download a Google Drive file, handling the confirm token used for large files."""
    file_id = _extract_gdrive_file_id(url)
    if not file_id:
        print(f"\n  ❌ Could not extract Google Drive file id for {desc}")
        return False

    base = "https://drive.google.com/uc?export=download"
    initial_url = f"{base}&id={urllib.parse.quote(file_id)}"

    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))

    def _stream_to_file(download_url: str) -> tuple[bool, str | None]:
        req = urllib.request.Request(download_url, headers={"User-Agent": "Mozilla/5.0"})
        with opener.open(req, timeout=60) as resp:
            content_type = resp.headers.get("Content-Type", "")
            # For the confirm-token page, Google returns text/html
            if "text/html" in content_type.lower():
                html = resp.read(2 * 1024 * 1024).decode("utf-8", errors="ignore")
                # Virus scan warning pages provide a download form with hidden inputs.
                # We'll parse the form and build the real download URL.

                action_url: str | None = None
                m = re.search(r"<form[^>]+action=\"([^\"]+)\"", html)
                if m:
                    action_url = m.group(1).replace("&amp;", "&")

                fields: dict[str, str] = {}
                for name, value in re.findall(r"name=\"([^\"]+)\"\s+value=\"([^\"]*)\"", html):
                    fields[name] = value

                # Cookie-based token (download_warning*) sometimes exists too.
                if "confirm" not in fields:
                    for c in cookie_jar:
                        if c.name.startswith("download_warning"):
                            fields["confirm"] = c.value
                            break

                # Fallback: confirm=<token> embedded in HTML
                if "confirm" not in fields:
                    m = re.search(r"confirm=([0-9A-Za-z_\-]+)", html)
                    if m:
                        fields["confirm"] = m.group(1)

                if not fields.get("confirm"):
                    return False, None

                if not action_url:
                    action_url = "https://drive.google.com/uc"
                elif action_url.startswith("/"):
                    action_url = "https://drive.google.com" + action_url
                elif not action_url.startswith("http"):
                    action_url = "https://drive.google.com/" + action_url.lstrip("/")

                query = urllib.parse.urlencode(fields)
                return False, action_url + ("&" if "?" in action_url else "?") + query

            total_size = resp.headers.get("Content-Length")
            total = int(total_size) if total_size and total_size.isdigit() else None

            dest.parent.mkdir(parents=True, exist_ok=True)
            downloaded = 0
            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        percent = min(100, int(downloaded * 100 / total))
                        mb_downloaded = downloaded / (1024 * 1024)
                        mb_total = total / (1024 * 1024)
                        print(
                            f"\r     Progress: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)",
                            end="",
                            flush=True,
                        )
            print()
            return True, None

    print(f"  📥 Downloading {desc} (Google Drive)...")
    print(f"     Source: {url}")
    print(f"     Destination: {dest}")

    ok, next_url = _stream_to_file(initial_url)
    if ok:
        return True

    if not next_url:
        print("\n  ❌ Google Drive download required confirmation but it could not be parsed (permission/quota?).")
        return False

    ok2, next_url2 = _stream_to_file(next_url)
    if ok2:
        return True

    if next_url2:
        # One more hop (rare), try once.
        ok3, _ = _stream_to_file(next_url2)
        return ok3
    return False


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download a file with progress indication."""
    if not url:
        print(f"  ⚠️  No URL configured for {desc}")
        return False

    if "drive.google.com" in url:
        return _download_google_drive(url, dest, desc)

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
        help="Output directory for weights (default: <project>/models)",
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

    args = parser.parse_args()

    if args.list:
        list_models()
        return 0

    # Determine weights directory
    if args.output_dir:
        weights_dir = args.output_dir.resolve()
    else:
        weights_dir = get_project_root() / "models"

    print("=" * 60)
    print("  MOT Model Weights Downloader")
    print("=" * 60)
    print(f"\n📁 Weights directory: {weights_dir}")

    # Determine which models to download
    models_to_download = args.models

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
