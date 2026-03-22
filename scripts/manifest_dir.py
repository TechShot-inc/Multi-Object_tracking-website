#!/usr/bin/env python3
"""Generate a deterministic manifest + checksums for a directory.

This is used for Triton model repo packaging (Phase 4 model lifecycle).

Outputs:
- JSON manifest with file list, sizes, and sha256
- sha256sums file compatible with `sha256sum -c`

Example:
  python scripts/manifest_dir.py --dir models_repo --out-json var/model_repo_manifest.json --out-sha256 var/model_repo.SHA256SUMS
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FileEntry:
    path: str
    size: int
    sha256: str


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file():
            files.append(p)
    # Deterministic ordering
    files.sort(key=lambda p: p.as_posix())
    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Directory to manifest")
    ap.add_argument("--out-json", required=True, help="Output JSON path")
    ap.add_argument("--out-sha256", required=True, help="Output SHA256SUMS path")
    ap.add_argument(
        "--root-prefix",
        default="",
        help="Optional prefix to prepend to paths in outputs (e.g. models_repo)",
    )
    args = ap.parse_args()

    root = Path(args.dir).resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    prefix = str(args.root_prefix).strip("/")

    entries: list[FileEntry] = []
    for f in _iter_files(root):
        rel = f.relative_to(root).as_posix()
        if prefix:
            rel = f"{prefix}/{rel}"
        st = f.stat()
        entries.append(FileEntry(path=rel, size=int(st.st_size), sha256=_sha256(f)))

    out_json = Path(args.out_json)
    out_sha = Path(args.out_sha256)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_sha.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": 1,
        "root": root.as_posix(),
        "file_count": len(entries),
        "files": [entry.__dict__ for entry in entries],
    }

    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # sha256sum format: "<sha256>  <path>"
    out_sha.write_text("".join(f"{e.sha256}  {e.path}\n" for e in entries), encoding="utf-8")

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_sha}")


if __name__ == "__main__":
    main()
