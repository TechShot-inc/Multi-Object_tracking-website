# Configuration & tuning

This document lists the important environment variables and explains how to tune quality vs speed.

## Where configuration lives

- The web app reads basic settings from env in `src/mot_web/config.py`.
- The realtime tracking stack reads most tuning knobs from env in `CustomBoostTrack/realtime_ensembling.py` and `src/mot_web/realtime_backend_app.py`.
- Docker Compose passes variables through in `docker/compose.yml`.

## Core app settings

These affect the FastAPI web app.

- `ENV`: `dev` (reload) or `prod`.
- `HOST`, `PORT`: bind host/port (default Docker uses `PORT=5000`).
- `PROJECT_ROOT`: base directory inside container.
- `UPLOAD_DIR`, `RESULTS_DIR`: where jobs and results go.
- `MAX_UPLOAD_MB`: max upload size.
- `QUEUE_MODE`: `inline` or `rq`.
- `REDIS_URL`: required when `QUEUE_MODE=rq`.

## Detector backend selection

- `DETECTOR_BACKEND`:
  - `local`: run Ultralytics models in the process.
  - `triton`: call Triton over gRPC.

When using Triton:

- `TRITON_URL`: e.g. `triton:8001`
- `TRITON_YOLO11_MODEL`: default `yolo11`
- `TRITON_YOLO12_MODEL`: default `yolo12`
- `TRITON_INPUT_SIZE`: input size expected by the model (default in code is 640 unless overridden).

## Realtime quality knobs (most important)

There are two “confidence thresholds” you should understand:

### 1) Per-model detector confidence

- `REALTIME_YOLO_CONF`

This threshold is applied per model before boxes enter fusion/tracking.

Notes:

- For Triton exports, confidence values can be on a different scale than local Ultralytics.
- Keeping `REALTIME_YOLO_CONF` too high can cause the detector to return almost no boxes.

### 2) Final fused-box confidence (recommended knob)

- `REALTIME_ENSEMBLE_CONF`

This is the main “reduce false boxes” control because it filters the WBF fused output.

Recommended tuning flow:

1) Set `REALTIME_ENSEMBLE_CONF` first (e.g. 0.3 → 0.5 → 0.7).
2) Only if you still see noise, raise `REALTIME_YOLO_CONF` slightly.

In Docker Compose, a default is provided. You can override it without editing files:

```bash
export REALTIME_ENSEMBLE_CONF=0.7
docker compose -f docker/compose.yml up -d --no-deps --build mot-realtime
```

### IoU / fusion behavior

- `REALTIME_ENSEMBLE_IOU`: WBF IoU threshold.

## Realtime performance knobs

### Frontend (browser)

The realtime UI supports URL query parameters to tune capture rate/size/quality.

Examples:

- `?rtFps=10` lower send FPS
- `?rtJpegQ=0.6` lower JPEG quality
- `?rtSendW=640&rtSendH=480` resize before sending

(See `src/mot_web/web/static/js/realtime.js`.)

### Backend preview encoding

The realtime backend returns an annotated JPEG preview. You can reduce CPU and bandwidth:

- `REALTIME_PREVIEW_MAX_W` (default 640)
- `REALTIME_PREVIEW_JPEG_QUALITY` (default 60)

### Timing instrumentation

- `REALTIME_TIMING=1` adds `timing_ms` to each WS metadata message.

### Triton concurrency (GPU memory)

- `REALTIME_TRITON_PARALLEL`: when `1`, YOLO11 and YOLO12 inference calls may overlap.

Notes:

- This can improve throughput, but on some GPUs it can trigger Triton/ONNXRuntime CUDA OOM.
- If you see Triton errors like `CUDA failure 2: out of memory` or `SafeInt... Integer overflow`, set `REALTIME_TRITON_PARALLEL=0` and restart `mot-realtime` (and restart `mot-triton` once to clear the error state).

## WBF workload reduction (still ensemble)

To keep the ensemble but reduce fusion cost:

- `REALTIME_WBF_TOPK`: top-K boxes per model before WBF (default 200)
- `REALTIME_WBF_SKIP_BOX_THR`: prefilter scores before WBF

## Safe defaults and debugging

- `REALTIME_DEBUG=1`: enables debug logging on hot paths.

## Practical presets

### Reduce false positives

- `REALTIME_ENSEMBLE_CONF=0.7`
- Optionally: `REALTIME_WBF_SKIP_BOX_THR=0.35`

### Maximize realtime responsiveness

- Use Triton GPU backend.
- Lower frontend FPS and JPEG quality.
- Keep preview small: `REALTIME_PREVIEW_MAX_W=640` or `480`.
