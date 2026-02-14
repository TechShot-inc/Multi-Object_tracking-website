# Multi-Object Tracking Web App (FastAPI + Docker + BoostTrack + Triton)

This repository is a production-oriented **FastAPI** web application for multi-object tracking, shipped with **Docker Compose**.

It supports two user-facing flows:

1) **Video upload (offline)**: upload a video, run the detection + tracking pipeline, then download results.
2) **Realtime (webcam)**: your browser streams frames over WebSocket; the backend runs detection + tracking and returns an annotated preview.

Key components:

- Web UI + API: FastAPI app (`mot-web`) serving templates + static assets and exposing HTTP + WebSocket routes.
- Tracking pipeline: BoostTrack-based tracker + YOLO detectors.
- Optional Triton inference: GPU-backed ensemble detector (YOLO11 + YOLO12) served via NVIDIA Triton Inference Server.

If you’re new to the project, start here:

- Docs index: [docs/index.md](docs/index.md)
- Development: [docs/development.md](docs/development.md)
- Architecture / system design: [docs/architecture.md](docs/architecture.md)
- API reference (HTTP + WebSocket): [docs/api.md](docs/api.md)
- Configuration & tuning (confidence thresholds, performance knobs): [docs/configuration.md](docs/configuration.md)
- FAQ: [docs/faq.md](docs/faq.md)
- Triton model workflow: [docs/triton_from_pt.md](docs/triton_from_pt.md)
- Troubleshooting: [docs/troubleshooting.md](docs/troubleshooting.md)

---

## Quickstart (Docker)

CPU-only (fast to get running):

1) Put weights under `models/` (or set env vars). Minimum:
   - `models/yolo11x.pt`
   - `models/yolo12x.pt`
   - `models/osnet_ain_ms_m_c.pth.tar`

2) Start the web + realtime stack:

`docker compose -f docker/compose.yml up -d --build`

3) Open:

`http://localhost:5000`

---

## Triton + GPU (recommended for realtime)

You’ll get the best realtime latency when the detector backend is Triton.

1) Ensure NVIDIA Container Toolkit works:

`docker run --rm --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi`

2) Start Triton profile:

`DETECTOR_BACKEND=triton docker compose -f docker/compose.yml -f docker/compose.gpu.yml --profile triton up -d --build`

3) Verify:

- Web UI: `http://localhost:5000`
- Triton ready: `http://localhost:8000/v2/health/ready`

---

## Realtime confidence tuning (common request)

To reduce false boxes, raise the *final fused-box threshold*:

- `REALTIME_ENSEMBLE_CONF` (default is set in [docker/compose.yml](docker/compose.yml))

Example:

`export REALTIME_ENSEMBLE_CONF=0.7`

Then recreate `mot-realtime` or restart the stack.

More knobs and how they interact: [docs/configuration.md](docs/configuration.md)

---

## Project entrypoints

- Web app (dev/prod): [src/mot_web/__main__.py](src/mot_web/__main__.py)
- App factory / router wiring: [src/mot_web/app_factory.py](src/mot_web/app_factory.py)
- Web routes:
  - [src/mot_web/web/routes/video.py](src/mot_web/web/routes/video.py)
  - [src/mot_web/web/routes/realtime.py](src/mot_web/web/routes/realtime.py)
  - [src/mot_web/web/routes/health.py](src/mot_web/web/routes/health.py)
- Dedicated realtime backend (for WebSocket proxy): [src/mot_web/realtime_backend_app.py](src/mot_web/realtime_backend_app.py)
- BoostTrack + ensemble logic: [CustomBoostTrack](CustomBoostTrack)

---

## License

See [CustomBoostTrack/LICENSE](CustomBoostTrack/LICENSE) for the tracking submodule’s license.

