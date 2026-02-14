# Triton models from `.pt` weights (YOLO)

Triton can’t serve Ultralytics `.pt` weights directly. The common path is:

`.pt` → ONNX → Triton model repository → (optional) publish as OCI artifact → run Triton + worker.

This repo already has:
- Triton init-container (`model-init`) that pulls a model repo OCI artifact into `/models`
- a Triton integration test profile (`triton-itest`) that checks server + model readiness

## 1) Export ONNX from `.pt`

Requires the ML deps:
- local: install `mot-web[ml]`
- or run inside the provided Docker targets (recommended for reproducibility)

Local export (example):

```bash
python scripts/export_yolo_pt_to_onnx.py \
  --weights models/yolo11x.pt \
  --imgsz 640 \
  --out artifacts/onnx/yolo11.onnx

python scripts/export_yolo_pt_to_onnx.py \
  --weights models/yolo12x.pt \
  --imgsz 640 \
  --out artifacts/onnx/yolo12.onnx
```

Tips:
- keep `--dynamic=false` initially; fixed shapes are easier for Triton + TensorRT later
- default export is **raw outputs** (no integrated NMS)

## 2) Create Triton model repository

This generates `config.pbtxt` using ONNX I/O names and writes the Triton layout:

```bash
python scripts/make_triton_model_repo.py \
  --onnx artifacts/onnx/yolo11.onnx \
  --model-name yolo11 \
  --repo-out models_repo \
  --max-batch-size 0 \
  --force

python scripts/make_triton_model_repo.py \
  --onnx artifacts/onnx/yolo12.onnx \
  --model-name yolo12 \
  --repo-out models_repo \
  --max-batch-size 0 \
  --force

Notes:
- If you export ONNX with a **dynamic batch dimension** (e.g. `--dynamic`), you can use `--max-batch-size 1`.
- If your ONNX input shape is fixed as `[1,3,640,640]`, use `--max-batch-size 0`.
```

You should now have:

- `models_repo/yolo11/1/model.onnx`
- `models_repo/yolo11/config.pbtxt`
- `models_repo/yolo12/1/model.onnx`
- `models_repo/yolo12/config.pbtxt`

## 3) Publish model repo as OCI artifact (for `model-init`)

This repo ships a helper script:

```bash
# Example (GHCR):
# bash scripts/publish_triton_model_repo_oci.sh ghcr.io/<you>/mot-triton-models:yolo11-yolo12 models_repo

bash scripts/publish_triton_model_repo_oci.sh <oci-ref> models_repo
```

It will print an OCI reference you can put in `TRITON_MODEL_REPO_REF`.

If you don’t have a registry yet:
- use GHCR (recommended)
- or run a local registry (`registry:2`) and push there

For a local HTTP registry (no TLS), set:

```bash
export ORAS_PLAIN_HTTP=1
```

## 4) Run Triton integration test profile

```bash
export TRITON_MODEL_REPO_REF='ghcr.io/<you>/<repo>@sha256:<digest>'
export ORAS_PLAIN_HTTP=0

docker compose -f docker/compose.yml --profile triton-itest up \
  --abort-on-container-exit --exit-code-from mot-triton-itests mot-triton-itests
```

If this passes, Triton is up and both models are discoverable.

## 5) GPU run on a laptop (RTX 4060)

Requirements on the host:
- NVIDIA driver installed
- NVIDIA Container Toolkit configured (`docker run --gpus all ...` must work)

### Fedora 43 host setup (NVIDIA Container Toolkit)

On Fedora, the driver (`nvidia-smi`) can be working while Docker still can’t see the GPU. You need the NVIDIA Container Toolkit.

1) Install toolkit + configure Docker (requires `sudo`):

```bash
sudo dnf install -y dnf-plugins-core
sudo dnf config-manager addrepo --from-repofile \
  https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo

sudo dnf install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

2) Verify Docker GPU access:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi
```

If this fails with `could not select device driver "" with capabilities: [[gpu]]`, Docker is not configured for GPU yet.

Then:

```bash
export DETECTOR_BACKEND=triton
export TRITON_MODEL_REPO_REF='ghcr.io/<you>/<repo>@sha256:<digest>'
export ORAS_PLAIN_HTTP=0

docker compose -f docker/compose.yml -f docker/compose.gpu.yml --profile triton up -d --build
```

Or use the helper:

```bash
export TRITON_MODEL_REPO_REF='ghcr.io/<you>/<repo>@sha256:<digest>'
./scripts/run_triton_stack.sh --gpu
```

At that point:
- `mot-web-dev` serves UI on `http://localhost:5000`
- `mot-worker` uses GPU and calls Triton over gRPC

## Troubleshooting

- If Triton starts but models aren’t ready, check `config.pbtxt` input/output names vs the ONNX graph.
- If detections look wrong, try exporting with a different `imgsz` and keep it consistent with `TRITON_INPUT_SIZE`.
- If you want maximum FPS, convert ONNX → TensorRT and serve `.plan` instead of ONNX.
