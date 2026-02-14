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
