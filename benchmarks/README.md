# Benchmarks (Docker-only)

This folder contains lightweight performance benchmarks intended to run **entirely inside Docker**.

## Realtime HTTP benchmark (synthetic frames)

What it measures
- End-to-end request latency of `POST /realtime/track` using deterministic synthetic frames.
- Reports $p50$, $p95$, mean latency, and effective FPS.

What it does *not* guarantee
- This benchmark is not a model-accuracy test.
- If the tracker/model weights are missing, the web app may operate in a "stub" mode (still useful for catching regressions in the request/encode path).

### Local run (Docker)

1) Start a web container that does **not** proxy realtime to the GPU backend:
- `docker compose -f docker/compose.yml --profile prod up -d --build mot-web-prod`

2) Run the benchmark **in Docker**, sharing the web container network namespace:
- `docker compose -f docker/compose.yml build mot-tests`
- `docker run --rm --network container:mot-web-prod mot-tests python scripts/benchmark_realtime.py --frames 200 --warmup 20 --out - > var/benchmark.json`

3) Shut down:
- `docker compose -f docker/compose.yml --profile prod down`

### Output

The benchmark prints a JSON document to stdout (when `--out -`).
It includes:
- `frames`, `warmup`, `width`, `height`, `jpeg_quality`
- `latency_ms`: `mean`, `p50`, `p95`, `min`, `max`
- `fps` and `total_s`
- `timestamp_ms`
- `git_sha` (from `GITHUB_SHA` if present)
