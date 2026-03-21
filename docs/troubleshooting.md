# Troubleshooting

This is the practical “when something breaks” guide.

## Web UI is up but realtime says tracker inactive

Symptoms:

- UI shows `tracker_active: false`
- Error mentions missing weights or missing module

Checks:

1) Verify weights are present inside `mot-realtime`:

```bash
docker exec mot-realtime ls -la /app/models
```

2) Verify the env vars:

```bash
docker exec mot-realtime printenv | egrep 'YOLO11_MODEL_PATH|YOLO12_MODEL_PATH|REID_MODEL_PATH|MODELS_DIR'
```

3) If you see `No module named 'CustomBoostTrack'`:

- Ensure the Compose bind mount exists and the container has `/app/CustomBoostTrack`.

## Realtime is laggy / delayed

Use `timing_ms` to find where time is spent:

- `decode` high: sending huge frames, increase compression or reduce resolution.
- `track` high: detector backend is slow (CPU), or Triton is not being used.
- `encode` high: lower `REALTIME_PREVIEW_JPEG_QUALITY` or `REALTIME_PREVIEW_MAX_W`.

Checklist:

1) Ensure Triton is enabled:

- `DETECTOR_BACKEND=triton` in `mot-realtime`

2) Ensure Triton is ready:

- `curl http://localhost:8000/v2/health/ready`

3) Ensure realtime calls Triton:

- `TRITON_URL=triton:8001`

## Triton container keeps restarting

Common causes:

- Model repo contains a directory that is not a model but looks like one.

Fix:

- Ensure the model repo volume contains only model folders (e.g. `yolo11/`, `yolo12/`).

## Triton returns OOM

- GPU VRAM is insufficient for two large models plus workspace.

Mitigations:

- Avoid having multiple containers allocate GPU VRAM unnecessarily.
- Reduce input size (`TRITON_INPUT_SIZE`) with a matching ONNX export.
- Reduce concurrent inference (disable parallel inference) while keeping the ensemble:

```bash
export REALTIME_TRITON_PARALLEL=0
```

## Too many wrong boxes

Primary knob:

- `REALTIME_ENSEMBLE_CONF` (increase to reduce false positives)

Secondary:

- `REALTIME_YOLO_CONF` (increase slightly if needed)
- increase `REALTIME_MIN_BOX_AREA` if tiny boxes are noise

## Port conflicts

Default ports:

- Web UI: `5000`
- Realtime backend (internal): `5001`
- Triton HTTP/gRPC/metrics: `8000/8001/8002`

If `5000` is in use, stop the conflicting process or change the `PORT` mapping.

## Browser console shows WebSocket errors but tracking still works

Common messages:

- `Unchecked runtime.lastError: Could not establish connection. Receiving end does not exist.`
- `Unchecked runtime.lastError: The message port closed before a response was received.`
- `reload.js:... WebSocket connection to 'ws://127.0.0.1:5500//ws' failed`
- `realtime.js ... Error: WebSocket timeout`

What they usually mean:

- The `runtime.lastError` messages are typically from a browser extension (not the app).
- The `reload.js` WebSocket to port `5500` is typically injected by a dev server like VS Code “Live Server/Live Preview”. It is not the app’s realtime socket.

How to confirm the app WebSocket works:

1) Open the realtime page from the FastAPI server (not a static HTML preview):
	- `http://localhost:5000/realtime`

2) In the browser DevTools Network tab, you should see a WebSocket to:
	- `ws://localhost:5000/realtime/ws` (or `wss://...` if you are on HTTPS)

If you instead open `realtime.html` via a static server on port `5500`, the frontend will try to connect to `ws://127.0.0.1:5500/realtime/ws` and will time out.
