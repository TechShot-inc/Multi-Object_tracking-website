#!/usr/bin/env python3
"""Create a GitHub release and upload assets (stdlib only).

Environment:
  GITHUB_TOKEN         (required)
  GITHUB_REPOSITORY    owner/repo (required)

Usage:
  python scripts/github_release_upload.py \
    --tag model-bundle-v1 \
    --name "Model bundle v1" \
    --notes "..." \
    --target main \
    --asset var/model_bundle/triton-models.tar.gz \
    --asset var/model_bundle/triton-models.tar.gz.sha256 \
    --asset var/model_bundle/model_repo_manifest.json

Notes:
- If the tag does not exist, GitHub will create it from --target.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def _api_request(url: str, *, method: str, token: str, data: dict | None = None, content_type: str | None = None):
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "User-Agent": "mot-web-model-release",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    body = None
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = content_type or "application/json"

    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
            return resp.status, resp.headers, raw
    except urllib.error.HTTPError as e:
        raw = e.read()
        raise SystemExit(f"GitHub API error {e.code}: {raw[:2000].decode('utf-8', errors='ignore')}")


def _upload_asset(upload_url_template: str, *, token: str, file_path: Path) -> None:
    # upload_url is like: https://uploads.github.com/repos/{owner}/{repo}/releases/{id}/assets{?name,label}
    upload_url = upload_url_template.split("{", 1)[0]
    q = urllib.parse.urlencode({"name": file_path.name})
    upload_url = f"{upload_url}?{q}"

    mime, _ = mimetypes.guess_type(file_path.name)
    if not mime:
        mime = "application/octet-stream"

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "User-Agent": "mot-web-model-release",
        "Content-Type": mime,
        "X-GitHub-Api-Version": "2022-11-28",
    }

    data = file_path.read_bytes()
    req = urllib.request.Request(upload_url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            _ = resp.read()
            if resp.status not in (200, 201):
                raise SystemExit(f"Asset upload failed: {file_path} status={resp.status}")
    except urllib.error.HTTPError as e:
        raw = e.read()
        raise SystemExit(
            f"Asset upload failed: {file_path} status={e.code}: {raw[:2000].decode('utf-8', errors='ignore')}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--notes", default="")
    ap.add_argument("--target", default="main", help="target_commitish")
    ap.add_argument("--prerelease", action="store_true")
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--asset", action="append", default=[])
    args = ap.parse_args()

    token = os.environ.get("GITHUB_TOKEN", "").strip()
    repo = os.environ.get("GITHUB_REPOSITORY", "").strip()
    if not token:
        raise SystemExit("GITHUB_TOKEN is required")
    if not repo or "/" not in repo:
        raise SystemExit("GITHUB_REPOSITORY is required (owner/repo)")

    owner, name = repo.split("/", 1)

    create_url = f"https://api.github.com/repos/{owner}/{name}/releases"
    payload = {
        "tag_name": args.tag,
        "name": args.name,
        "body": args.notes,
        "target_commitish": args.target,
        "draft": bool(args.draft),
        "prerelease": bool(args.prerelease),
    }

    status, _headers, raw = _api_request(create_url, method="POST", token=token, data=payload)
    if status not in (200, 201):
        raise SystemExit(f"Release creation failed (status={status})")

    resp = json.loads(raw.decode("utf-8"))
    upload_url = resp.get("upload_url")
    html_url = resp.get("html_url")
    if not upload_url:
        raise SystemExit("Release created but upload_url missing")

    assets = [Path(p) for p in args.asset]
    for p in assets:
        if not p.exists() or not p.is_file():
            raise SystemExit(f"Asset not found: {p}")

    for p in assets:
        print(f"Uploading asset: {p}")
        _upload_asset(upload_url, token=token, file_path=p)

    print(f"Release ready: {html_url}")


if __name__ == "__main__":
    main()
