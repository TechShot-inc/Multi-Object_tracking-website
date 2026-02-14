# Architecture & system design

This document explains the system at a component level: which containers exist, how traffic flows, and where the tracking pipeline lives.

## High-level diagram

```mermaid
flowchart LR
  B[Browser\n(realtime.html)] -->|WS JPEG frames| W[Web App\nmot-web-dev\nFastAPI]
  W -->|WS proxy| R[Realtime Backend\nmot-realtime\nFastAPI WS]
  R -->|gRPC| T[Triton Inference Server\nmot-triton]

  B -->|HTTP upload| W
  W -->|inline or RQ| P[Pipeline\nrun_video]
  P -->|optional queue| Q[Redis + RQ]

  subgraph Storage
    U[(uploads volume)]
    S[(results volume)]
  end

  W --> U
  P --> S
```

## Components

### Web application (`mot-web`)

- Serves HTML templates and static JS/CSS.
- Exposes the HTTP API for video upload and job status.
- Exposes the `/realtime/ws` WebSocket endpoint.

In production compose, `/realtime/ws` is a proxy to the dedicated realtime backend.

Code:

- App wiring: `src/mot_web/app_factory.py`
- Routes: `src/mot_web/web/routes/*`

### Realtime backend (`mot-realtime`)

- Dedicated FastAPI app for realtime WebSocket tracking.
- Runs a “one tracker per WebSocket connection” model for isolation.
- Receives JPEG bytes, decodes to BGR, runs tracking, draws overlays, and returns:
  1) JSON metadata
  2) annotated JPEG bytes

Code:

- `src/mot_web/realtime_backend_app.py`
- BoostTrack glue: `CustomBoostTrack/realtime_ensembling.py`

### Triton (`mot-triton`)

- NVIDIA Triton Inference Server.
- Serves the YOLO11 + YOLO12 ONNX models.
- Realtime calls Triton over gRPC via `tritonclient[grpc]`.

Model repo layout:

- `models_repo/yolo11/1/model.onnx`
- `models_repo/yolo11/config.pbtxt`
- `models_repo/yolo12/1/model.onnx`
- `models_repo/yolo12/config.pbtxt`

How to export/publish models:

- `docs/triton_from_pt.md`

## Data flows

### Realtime (browser)

1) Browser captures a frame from the camera.
2) Browser sends a JPEG-encoded frame to `/realtime/ws`.
3) Web app proxies the WebSocket to `mot-realtime`.
4) `mot-realtime`:
   - decodes JPEG
   - runs detector ensemble (YOLO11 + YOLO12) + WBF
   - runs BoostTrack update
   - draws boxes + IDs + optional ROI/line
   - downscales preview (optional) and encodes JPEG
5) Backend sends JSON metadata then JPEG bytes back.

### Offline / video upload

1) Client uploads a video to `/video/upload`.
2) Client polls `/video/status/{job_id}`.
3) Client triggers `/video/run/{job_id}`.
4) Pipeline writes artifacts to the job folder under results.

Queue modes:

- `QUEUE_MODE=inline`: run in web process.
- `QUEUE_MODE=rq`: enqueue in Redis for `mot-worker`.

## Design choices

### Why WebSocket for realtime

- Lower overhead than HTTP per frame.
- Allows strict pacing/backpressure.

### One tracker per WebSocket

- Each browser session has its own tracker state.
- Prevents cross-user ID bleed.

### Ensemble detector

- Two detectors (YOLO11 + YOLO12) produce candidate boxes.
- Weighted Box Fusion (WBF) merges overlapping boxes.
- A final confidence threshold filters fused boxes.

The ensemble is intentionally preserved even during performance tuning.

## Performance model

Realtime latency is usually dominated by:

- detector inference (Triton or local)
- fusion cost (WBF can be expensive with many boxes)

To help diagnose, the realtime backend can include per-stage timings (`timing_ms`).

See tuning knobs in `docs/configuration.md`.
