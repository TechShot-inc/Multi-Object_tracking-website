from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any

import httpx

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False


@dataclass(frozen=True)
class BenchmarkStats:
    frames: int
    warmup: int
    width: int
    height: int
    jpeg_quality: int
    total_s: float
    fps: float
    latency_ms: dict[str, float]
    timestamp_ms: int
    base_url: str
    endpoint: str
    git_sha: str


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    if pct <= 0:
        return min(values)
    if pct >= 100:
        return max(values)
    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(values_sorted) - 1)
    if f == c:
        return values_sorted[f]
    d0 = values_sorted[f] * (c - k)
    d1 = values_sorted[c] * (k - f)
    return d0 + d1


def _make_deterministic_frame(width: int, height: int, frame_index: int) -> "np.ndarray":
    # Deterministic synthetic RGB image with moving shapes.
    img = np.zeros((height, width, 3), dtype=np.uint8)
    t = frame_index % 255
    img[:, :, 0] = (np.arange(width, dtype=np.uint16)[None, :] + t) % 256
    img[:, :, 1] = (np.arange(height, dtype=np.uint16)[:, None] + 2 * t) % 256
    img[:, :, 2] = (t * 3) % 256

    # Add a moving rectangle + text for a bit of texture.
    x1 = int((frame_index * 7) % max(1, width - 60))
    y1 = int((frame_index * 5) % max(1, height - 40))
    cv2.rectangle(img, (x1, y1), (x1 + 60, y1 + 40), (255, 255, 255), 2)
    cv2.putText(img, f"{frame_index}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return img


def _encode_jpeg_bytes(frame_bgr: "np.ndarray", jpeg_quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def run_http_benchmark(
    *,
    base_url: str,
    endpoint: str,
    frames: int,
    warmup: int,
    width: int,
    height: int,
    jpeg_quality: int,
    timeout_s: float,
    wait_s: float,
) -> BenchmarkStats:
    if not CV2_AVAILABLE:
        raise RuntimeError("OpenCV/numpy not available in this environment")

    url = base_url.rstrip("/") + endpoint
    health_url = base_url.rstrip("/") + "/health"

    # A simple line config so we exercise that path, but this is not a correctness test.
    line_json = json.dumps({"position": "left", "x": 0.5})

    latencies_ms: list[float] = []

    git_sha = os.getenv("GITHUB_SHA", "unknown")

    started = time.perf_counter()
    with httpx.Client(timeout=httpx.Timeout(timeout_s)) as client:
        # Wait for the server to be ready (common when containers are just started).
        wait_deadline = time.perf_counter() + max(0.0, float(wait_s))
        last_err: str | None = None
        while True:
            try:
                r = client.get(health_url)
                if r.status_code == 200:
                    break
                last_err = f"/health returned HTTP {r.status_code}"
            except Exception as e:
                last_err = str(e)

            if time.perf_counter() >= wait_deadline:
                raise RuntimeError(f"Server not ready after {wait_s}s: {last_err}")
            time.sleep(0.5)

        for i in range(frames + warmup):
            frame = _make_deterministic_frame(width, height, i)
            jpeg_bytes = _encode_jpeg_bytes(frame, jpeg_quality)

            files = {"frame": ("frame.jpg", jpeg_bytes, "image/jpeg")}
            data = {"line": line_json}

            t0 = time.perf_counter()
            resp = client.post(url, files=files, data=data)
            t1 = time.perf_counter()

            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")

            # Ensure the server actually returned JSON.
            _ = resp.json()

            if i >= warmup:
                latencies_ms.append((t1 - t0) * 1000.0)

    total_s = time.perf_counter() - started

    if not latencies_ms:
        raise RuntimeError("No benchmark samples collected")

    latency_ms = {
        "mean": float(statistics.fmean(latencies_ms)),
        "p50": float(_percentile(latencies_ms, 50)),
        "p95": float(_percentile(latencies_ms, 95)),
        "min": float(min(latencies_ms)),
        "max": float(max(latencies_ms)),
    }

    fps = float(frames / max(total_s, 1e-9))

    return BenchmarkStats(
        frames=frames,
        warmup=warmup,
        width=width,
        height=height,
        jpeg_quality=jpeg_quality,
        total_s=float(total_s),
        fps=fps,
        latency_ms=latency_ms,
        timestamp_ms=int(time.time() * 1000),
        base_url=base_url,
        endpoint=endpoint,
        git_sha=git_sha,
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Realtime /realtime/track benchmark (Docker-only friendly)")
    p.add_argument(
        "--base-url",
        default="http://127.0.0.1:5000",
        help="Base URL reachable from inside Docker (recommended with --network container:mot-web-prod)",
    )
    p.add_argument("--endpoint", default="/realtime/track", help="HTTP endpoint path")
    p.add_argument("--frames", type=int, default=200, help="Measured frames")
    p.add_argument("--warmup", type=int, default=20, help="Warmup frames (not included in stats)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--jpeg-quality", type=int, default=80)
    p.add_argument("--timeout-s", type=float, default=30.0)
    p.add_argument("--wait-s", type=float, default=30.0, help="Wait up to this many seconds for /health before starting")
    p.add_argument("--out", default="-", help="Output path for JSON, or '-' for stdout")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)

    stats = run_http_benchmark(
        base_url=args.base_url,
        endpoint=args.endpoint,
        frames=args.frames,
        warmup=args.warmup,
        width=args.width,
        height=args.height,
        jpeg_quality=args.jpeg_quality,
        timeout_s=args.timeout_s,
        wait_s=args.wait_s,
    )

    payload: dict[str, Any] = {
        "frames": stats.frames,
        "warmup": stats.warmup,
        "width": stats.width,
        "height": stats.height,
        "jpeg_quality": stats.jpeg_quality,
        "total_s": stats.total_s,
        "fps": stats.fps,
        "latency_ms": stats.latency_ms,
        "timestamp_ms": stats.timestamp_ms,
        "base_url": stats.base_url,
        "endpoint": stats.endpoint,
        "git_sha": stats.git_sha,
    }

    out = args.out
    text = json.dumps(payload, indent=2, sort_keys=True)

    if out == "-":
        sys.stdout.write(text + "\n")
        return 0

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
