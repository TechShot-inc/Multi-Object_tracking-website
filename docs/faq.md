# FAQ

## Do you really keep the ensemble (YOLO11 + YOLO12)?

Yes. The design goal is to preserve a two-model ensemble and fuse results (WBF) even when optimizing performance.

If you need more speed, prefer:

- lowering the browser send FPS / resolution / JPEG quality
- reducing preview size (`REALTIME_PREVIEW_MAX_W`)
- reducing WBF workload (`REALTIME_WBF_TOPK`, `REALTIME_WBF_SKIP_BOX_THR`)

## Which confidence threshold should I change?

Use `REALTIME_ENSEMBLE_CONF` first.

Reason: it filters the **final fused boxes**, so it reduces false positives without prematurely removing per-model candidates.

Only after that, consider `REALTIME_YOLO_CONF`.

## Why does realtime behavior differ between Docker and local runs?

There are two different realtime implementations:

- Web app realtime (fallback): `src/mot_web/web/routes/realtime.py`
  - WS replies are JSON only and include a base64 JPEG (`annotated`).
- Dedicated realtime backend (Compose default): `src/mot_web/realtime_backend_app.py`
  - For each frame, replies with JSON metadata then a binary JPEG preview.

When `REALTIME_BACKEND_WS` is set, the web app proxies the browser WS to the dedicated backend.

## Why is realtime delayed / laggy?

Common causes:

- sending too many large frames (bandwidth + decode + encode)
- running detector inference on CPU (Triton disabled)
- WBF merging too many boxes

Suggested approach:

1) Enable `REALTIME_TIMING=1` and watch `timing_ms.track`.
2) Ensure `DETECTOR_BACKEND=triton` and Triton is ready.
3) Reduce browser send FPS and preview encoding cost.

## How do I confirm Triton is being used?

- In Docker, ensure `DETECTOR_BACKEND=triton` is set for `mot-realtime`.
- Check readiness: `curl http://localhost:8000/v2/health/ready`
- Tail realtime logs and look for Triton client initialization messages.

## Can I run without Redis / RQ?

Yes.

Set `QUEUE_MODE=inline` (default in most dev setups) to run video jobs in-process.
