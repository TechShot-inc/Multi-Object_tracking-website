System Design (Single NVIDIA GPU VM, fully containerized)
Services (Docker Compose)
web (FastAPI + UI, CPU-only)
Serves: /, /video/, /realtime/, REST APIs, and WebSocket realtime stream.
Never imports Torch/Ultralytics (keeps it small/fast).
realtime (FastAPI, GPU-enabled)
Owns realtime tracking state and inference.
Runs YOLO11+YOLO12 ensemble + BoostTrack per WebSocket session.
Web proxies browser WebSocket traffic to this service (same-origin UI, GPU-backed compute).
Note: realtime startup should not require COCO/TrackEval tooling (e.g. `pycocotools`) or YOLOX; those are only needed for offline dataset/eval paths.
worker (RQ worker, GPU-enabled)
Pulls jobs from Redis and runs the offline pipeline (BoostTrack/CustomBoostTrack).
Calls Triton for inference (YOLO11 + YOLO12), performs fusion (WBF) + tracking + renders artifacts.
redis (queue + lightweight state)
RQ broker + job status keys.
triton (NVIDIA Triton Inference Server, GPU-enabled)
Hosts TensorRT engines for YOLO11/YOLO12 (and later ReID if you want).
model-init (tiny init container, CPU-only)
Pulls the Triton “model repository” artifact (OCI) into a shared volume before Triton starts.

Compose wiring (implemented):
- Service: `model-init` (profile: `triton`)
- Env: `TRITON_MODEL_REPO_REF` (OCI ref, ideally pinned by digest)
- Volume: `triton-model-repo` mounted at `/models`
- `triton` depends on `model-init` with `service_completed_successfully`
Data / Artifact Contract (unchanged for the website)
Per job folder under RESULTS_DIR/{job_id}/:

meta.json, status.json
annotations.json (always produced; even on failure -> stub with error)
annotated_video.mp4 (best-effort; fallback to original upload still supported)
“Single env / dependencies” (efficient + deploy-anywhere)
You want minimal time/storage and “single env”. The only way to satisfy both is:

Single source of truth for dependencies (one lockfile)
Multiple runtime images (web vs worker) built from that lockfile, each installing only what it needs
This avoids the worst-case (one huge image everywhere) while still being “one environment definition”.

Recommended dependency tooling
Use uv (fast, reproducible) with:
pyproject.toml as the canonical spec
one uv.lock committed to the repo (implemented)
Dependency groups (still one lock)
In pyproject.toml, define groups:

web: fastapi, uvicorn[standard], jinja2, python-multipart, small utils
worker: adds torch (CUDA build), ultralytics, torchreid, opencv-python, ffmpeg bindings as needed, tritonclient, rq, etc.
dev: pytest
The lockfile pins exact versions for all groups so any machine builds reproducibly.

Triton design (YOLO11 + YOLO12 ensemble)
Model repo layout (volume-mounted into Triton)
models_repo/yolo11/1/model.plan + config.pbtxt
models_repo/yolo12/1/model.plan + config.pbtxt
Inference strategy (fast + simple)
Worker calls Triton twice per frame (yolo11 + yolo12)
Worker runs Weighted Box Fusion (already in your project deps) and then passes detections to BoostTrack
This gives you ensemble accuracy without Triton “ensemble models” complexity
Model build pipeline (one-time per version)
Export Ultralytics model → ONNX
Build TensorRT engine (.plan) with fixed input size + FP16
Package as Triton model repo folder
Publish that folder as an OCI artifact to your registry (Harbor)
Redis + RQ job execution (effective, minimal overhead)
Queue behavior
POST /video/run/{job_id} enqueues immediately (returns queued)
Worker consumes jobs sequentially (concurrency = 1 recommended on a single GPU)
Status is updated in:
Redis keys for quick polling and
status.json for UI compatibility and “filesystem is source-of-truth” recovery
Job state model
created → queued → running → done|failed
Store message, progress, updated_at_unix, and optional error
Realtime via WebSocket (efficient, GPU-backed)
API shape
WebSocket endpoint (browser-facing): /realtime/ws (proxied by web)
WebSocket endpoint (GPU backend): /realtime/ws (internal to docker network)
Client sends binary JPEG frames (or base64 if you must, but binary is better)
Server returns a small JSON metadata message (counts, timestamp, tracker_active) followed by binary JPEG (annotated)
Key design points
Backpressure: server should drop frames if it’s behind (keeps latency stable)
Tracker lifetime:
1 tracker instance per websocket connection (clean isolation)
or 1 shared tracker with per-client state (more complex; not recommended first)
Full Implementation Plan (every step, end-to-end)
Step 1 — Finalize FastAPI structure + remove remaining Flask coupling
Files to converge on FastAPI-only (no Flask leftovers):

app_factory.py
__main__.py
health.py
index.py
video.py
realtime.py
Templates already:
index.html
video.html
realtime.html
Concrete requirements:

Keep existing paths exactly (so your frontend JS keeps working)
Ensure /static/... works everywhere
Make /health a simple JSON response (for container healthchecks)
Step 2 — Add Redis + RQ “run job” flow
New module(s) (recommended):

src/mot_web/queue/redis.py (Redis connection + settings)
src/mot_web/queue/jobs.py (enqueue, status update helpers)
src/mot_web/worker/entrypoint.py (RQ worker startup)
Update video.py:
POST /video/run/{job_id} enqueues
GET /video/status/{job_id} checks Redis first, falls back to status.json
Worker behavior:

Reads UPLOAD_DIR/{filename}
Runs pipeline.py
Writes artifacts into RESULTS_DIR/{job_id}/...
Updates Redis + status.json
Step 3 — Integrate Triton inference into the pipeline
Add a detector abstraction inside your tracking layer:

TritonYoloDetector (calls Triton via tritonclient)
LocalUltralyticsDetector (fallback for dev; optional)
Where to wire:

pipeline.py
boosttrack_pipeline.py
Environment knobs:

DETECTOR_BACKEND=triton|local
TRITON_URL=triton:8001 (grpc) or :8000 (http)
Step 4 — WebSocket realtime endpoint + frontend wiring
Backend:

Add /ws/realtime in realtime.py
Keep /realtime/track as a compatibility fallback (useful for debugging)
Frontend:

Update realtime.js to:
prefer websocket
fallback to HTTP POST if websocket fails
Step 5 — Containerization optimized for speed + storage
Update Dockerfile into multi-stage builds:

web image:
Base: python:3.11-slim
Installs only web deps from lockfile
worker image:
Base: nvidia/cuda:<runtime>-ubuntu22.04 (runtime-only, not devel)
Installs worker deps (torch CUDA wheels + tritonclient + rq)
triton image:
Use NVIDIA NGC nvcr.io/nvidia/tritonserver:<version> (don’t build your own)
Update compose.yml:

web: ports 5000:5000, CPU only
redis: internal
worker: GPU reservation + volumes (uploads, results)
triton: GPU reservation + volume triton-model-repo
model-init: pulls model repo artifact → writes to triton-model-repo volume
Step 6 — Model distribution (fast deploy anywhere, no Google Drive in prod)
Keep download_models.py for dev only.

Production flow:

Build Triton repo → publish as OCI artifact
Deploy:
model-init pulls by digest pin to ensure reproducibility
Triton starts using that mounted repo
Step 7 — Testing “everything” (automated + container smoke)
Update tests to FastAPI TestClient:

test_routes.py
test_video_upload.py
test_video_run.py
test_video_results.py
test_integration.py
test_error_handling.py
test_realtime.py
Add new tests (to cover new architecture):

Queue tests: enqueue/run with a fake worker function (no GPU required)
WebSocket tests: connect, send 1 frame, receive annotated response (skip if OpenCV missing)
Triton integration tests: mocked client by default; real Triton only in an integration profile

Implementation notes (current repo):
- A real Redis/RQ integration test exists and is wired behind an optional compose profile:
	- Run: docker compose -f docker/compose.yml --profile itest run --rm mot-itests
	- This brings up redis and runs tests/test_queue_integration.py (in-process FastAPI TestClient + burst RQ worker) against it.
- A real Triton “server up + models loaded” integration test is wired behind an optional compose profile:
	- Requires: TRITON_MODEL_REPO_REF set to a published OCI artifact (see scripts/publish_triton_model_repo_oci.sh)
	- Run: docker compose -f docker/compose.yml --profile triton-itest up --abort-on-container-exit --exit-code-from mot-triton-itests mot-triton-itests
- Pytest has a timeout guard enabled (prevents hung tests from blocking indefinitely).
Manual container smoke checklist:

GET /health success
UI pages load (index/video/realtime)
Upload → enqueue → worker completes → results endpoints return artifacts
WebSocket realtime shows annotated frames
Efficiency / Minimal storage guarantees (what makes this deploy-anywhere)
Web container stays small (no Torch, no Ultralytics, no TensorRT)
Worker container is the only “heavy” one (and only deployed where GPU exists—which you said is always)
Triton is a standard NGC image (cacheable, stable)
Models are externalized into an OCI artifact (pulled once, cached in a volume), not baked into every image
Single lockfile ensures “works everywhere” reproducibility