# Development guide

This document is the “how to work on this repo” reference: local setup, Docker workflows, tests, and common dev loops.

## What you are building

This project is a FastAPI web application (templates + static UI) that runs a multi-object tracking pipeline.

Two main user flows:

- **Offline / video upload**: upload a video, run a tracking job, download annotated output and JSON artifacts.
- **Realtime / webcam**: browser streams JPEG frames over WebSocket; backend returns metadata + an annotated JPEG preview.

## Repo layout (high level)

- `src/mot_web/`: FastAPI app, routes, worker, queue, tracking pipeline glue.
- `CustomBoostTrack/`: tracking implementation and detector backends.
- `docker/`: Compose + Dockerfile targets.
- `models/`: local weights (mounted read-only into containers).
- `models_repo/`: Triton model repository (ONNXRuntime backend).
- `docs/`: project documentation.

## Prerequisites

### Docker (recommended)

- Docker Engine + Docker Compose v2.
- If using GPU/Triton: NVIDIA driver + NVIDIA Container Toolkit.

### Local Python (optional)

Local development is supported, but most contributors use Docker.

- Python >= 3.10
- `uv` (used by the Dockerfile)

## Running the app (Docker)

### 1) Put model weights in place

The ML containers expect weights under `models/` (or via env vars pointing elsewhere).

Minimum for local (non-Triton) detector:

- `models/yolo11x.pt`
- `models/yolo12x.pt`
- `models/osnet_ain_ms_m_c.pth.tar`

### 2) Start CPU stack

```bash
docker compose -f docker/compose.yml up -d --build
```

Open:

- Web UI: `http://localhost:5000`

### 3) Start Triton + GPU stack

```bash
DETECTOR_BACKEND=triton \
  docker compose -f docker/compose.yml -f docker/compose.gpu.yml --profile triton up -d --build
```

Health checks:

- Web UI: `http://localhost:5000`
- Triton HTTP ready: `http://localhost:8000/v2/health/ready`

Notes:

- `mot-web-dev` proxies `/realtime/ws` to `mot-realtime` when `REALTIME_BACKEND_WS` is set (Compose does this).
- When using Triton, realtime should rely on Triton for GPU inference; the `mot-realtime` container is intentionally configured to avoid stealing VRAM.

## Monitoring (Phase 5)

This repo includes a Docker Compose `monitoring` profile with Prometheus + Grafana.

Start monitoring stack:

```bash
docker compose -f docker/compose.yml --profile monitoring up -d
```

If you already have something using `3000` or `9090` on your host, override the published ports:

```bash
GRAFANA_PORT=3001 PROMETHEUS_PORT=9091 \
  docker compose -f docker/compose.yml --profile monitoring up -d
```

Open:

- Grafana: `http://localhost:${GRAFANA_PORT:-3000}` (default: `admin` / `admin`)
- Prometheus: `http://localhost:${PROMETHEUS_PORT:-9090}`

Metrics endpoints (inside the compose network):

- Web: `http://mot-web-dev:5000/metrics` (or `mot-web-prod:5000` when using `--profile prod`)
- Realtime backend: `http://mot-realtime:5001/metrics`
- Worker: `http://mot-worker:9101/metrics`
- Triton: `http://triton:8002/metrics`

## Common development loops

### Frontend iteration (JS/CSS)

- Static assets are served from `src/mot_web/web/static/`.
- In dev mode, static responses are set to `Cache-Control: no-store` to reduce stale asset issues.

### Backend iteration (FastAPI)

The `mot-web` entrypoint runs Uvicorn with reload when `ENV=dev`.

See:

- `src/mot_web/__main__.py`
- `src/mot_web/app_factory.py`

### Realtime tuning loop

Most realtime knobs are environment variables. You can set them and recreate `mot-realtime`.

Example (reduce false positives):

```bash
export REALTIME_ENSEMBLE_CONF=0.7
docker compose -f docker/compose.yml up -d --no-deps --build mot-realtime
```

See full list: `docs/configuration.md`.

## Tests

### Run unit/integration tests (Docker)

```bash
docker compose -f docker/compose.yml run --rm mot-tests
```

### Triton integration tests

There is a Triton profile test container.

```bash
export TRITON_MODEL_REPO_REF='ghcr.io/<you>/<repo>@sha256:<digest>'
DETECTOR_BACKEND=triton \
  docker compose -f docker/compose.yml --profile triton-itest up \
  --abort-on-container-exit --exit-code-from mot-triton-itests mot-triton-itests
```

## Code style and conventions

- Prefer small, targeted changes.
- Any new tunable should:
  - be wired via env vars in Compose when appropriate
  - have a default
  - be documented in `docs/configuration.md`

## Useful commands

- Tail logs: `docker logs -f mot-realtime`
- Recreate a service: `docker compose up -d --no-deps --build mot-realtime`
- Check env inside a container: `docker exec mot-realtime printenv | sort`

## Self-hosted GPU runner (on-demand)

If you want to run the GPU benchmark workflow on your own machine (laptop/desktop) without keeping it on 24/7, run the GitHub Actions runner only when needed.

Recommended option (runs exactly one job, then exits):

```bash
RUNNER_DIR=/mnt/Extra/actions-runner ./scripts/run_self_hosted_runner_once.sh
```

Alternative option (service you can start/stop manually):

```bash
cd /mnt/Extra/actions-runner
sudo ./svc.sh install
sudo ./svc.sh start
sudo ./svc.sh stop
sudo ./svc.sh status
```

Notes:

- The GPU benchmark workflow is manual (`workflow_dispatch`) and includes concurrency so only one GPU bench job runs at a time.
- GitHub Actions runs a workflow from a specific branch/SHA. If you click **Re-run jobs**, it reuses the old SHA (so it will not pick up workflow fixes). To pick up the latest workflow changes, use **Run workflow** and select the intended branch (typically `main`).
- The Triton GPU benchmark expects a Triton model repository on the runner host (default: `/mnt/Extra/mot-triton-model-repo`). You can change it in the workflow inputs when dispatching.
- Avoid using self-hosted runners for untrusted code (for example, auto-running fork PRs) since the runner has access to your machine.
