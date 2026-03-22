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
import http.client
import json
import mimetypes
import os
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


class GitHubAPIError(RuntimeError):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


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
        msg = raw[:2000].decode("utf-8", errors="ignore")
        raise GitHubAPIError(e.code, f"GitHub API error {e.code}: {msg}")


def _upload_asset(
    upload_url_template: str,
    *,
    token: str,
    file_path: Path,
    timeout_seconds: int,
    retries: int,
    chunk_size_bytes: int,
) -> None:
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

    parts = urllib.parse.urlsplit(upload_url)
    if parts.scheme != "https":
        raise SystemExit(f"Asset upload URL must be https (got {parts.scheme}): {upload_url}")

    path = parts.path
    if parts.query:
        path = f"{path}?{parts.query}"

    total_size = file_path.stat().st_size
    headers_with_length = dict(headers)
    headers_with_length["Content-Length"] = str(total_size)

    last_error: Exception | None = None
    for attempt in range(1, max(1, retries) + 1):
        conn: http.client.HTTPSConnection | None = None
        try:
            conn = http.client.HTTPSConnection(parts.netloc, timeout=timeout_seconds)
            conn.putrequest("POST", path)
            for k, v in headers_with_length.items():
                conn.putheader(k, v)
            conn.endheaders()

            with file_path.open("rb") as f:
                while True:
                    chunk = f.read(chunk_size_bytes)
                    if not chunk:
                        break
                    conn.send(chunk)

            resp = conn.getresponse()
            body = resp.read()
            if resp.status in (200, 201):
                return

            if resp.status == 422:
                detail = body[:2000].decode("utf-8", errors="ignore")
                # Common on reruns: asset with same name already exists.
                if "already_exists" in detail or "already exists" in detail:
                    print(f"Asset already exists; skipping: {file_path}")
                    return

            detail = body[:2000].decode("utf-8", errors="ignore")
            raise SystemExit(f"Asset upload failed: {file_path} status={resp.status}: {detail}")
        except (TimeoutError, socket.timeout, OSError) as e:
            last_error = e
            if attempt >= retries:
                break
            wait_seconds = min(60, 2**attempt)
            print(f"Upload attempt {attempt} failed ({type(e).__name__}: {e}); retrying in {wait_seconds}s...", file=sys.stderr)
            time.sleep(wait_seconds)
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    raise SystemExit(f"Asset upload failed after {retries} attempts: {file_path}: {last_error}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--notes", default="")
    ap.add_argument("--target", default="main", help="target_commitish")
    ap.add_argument("--prerelease", action="store_true")
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--asset", action="append", default=[])
    ap.add_argument("--upload-timeout-seconds", type=int, default=1800)
    ap.add_argument("--upload-retries", type=int, default=3)
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

    try:
        status, _headers, raw = _api_request(create_url, method="POST", token=token, data=payload)
        if status not in (200, 201):
            raise SystemExit(f"Release creation failed (status={status})")
        resp = json.loads(raw.decode("utf-8"))
    except GitHubAPIError as e:
        # Common on reruns: tag already exists -> reuse release.
        if e.code == 422 and ("already_exists" in str(e) or "already exists" in str(e) or "Validation Failed" in str(e)):
            get_url = f"https://api.github.com/repos/{owner}/{name}/releases/tags/{urllib.parse.quote(args.tag)}"
            status, _headers, raw = _api_request(get_url, method="GET", token=token)
            if status != 200:
                raise SystemExit(f"Release exists but lookup failed (status={status}): {e}")
            resp = json.loads(raw.decode("utf-8"))
            print(f"Release tag already exists; reusing: {args.tag}")
        else:
            raise SystemExit(str(e))

    upload_url = resp.get("upload_url")
    html_url = resp.get("html_url")
    if not upload_url:
        raise SystemExit("Release is missing upload_url")

    assets = [Path(p) for p in args.asset]
    for p in assets:
        if not p.exists() or not p.is_file():
            raise SystemExit(f"Asset not found: {p}")

    for p in assets:
        print(f"Uploading asset: {p}")
        _upload_asset(
            upload_url,
            token=token,
            file_path=p,
            timeout_seconds=max(30, int(args.upload_timeout_seconds)),
            retries=max(1, int(args.upload_retries)),
            chunk_size_bytes=8 * 1024 * 1024,
        )

    print(f"Release ready: {html_url}")


if __name__ == "__main__":
    main()
