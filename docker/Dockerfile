ARG BASE_IMAGE=python:3.11-slim
ARG CUDA_BASE_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Base OS layer shared by all targets
FROM ${BASE_IMAGE} AS base

LABEL maintainer="MOT Team"
LABEL description="Multi-Object Tracking Web Application"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

COPY pyproject.toml ./
COPY uv.lock ./
COPY src/ ./src/
COPY scripts/ ./scripts/

ARG PIP_TIMEOUT=120
ARG PIP_RETRIES=10


# -----------------
# Web target (no ML)
# -----------------
FROM base AS web

ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH=/app/.venv/bin:$PATH

RUN --mount=type=cache,target=/root/.cache/uv \
    python -m pip install --upgrade pip setuptools wheel uv && \
    uv sync --frozen

RUN mkdir -p /app/var/uploads /app/var/results && \
    chown -R appuser:appuser /app

USER appuser

ENV HOST=0.0.0.0 \
    PORT=5000 \
    ENV=prod \
    PROJECT_ROOT=/app \
    UPLOAD_DIR=/app/var/uploads \
    RESULTS_DIR=/app/var/results

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

CMD ["mot-web"]


# -----------------
# Worker target (ML)
# -----------------
FROM base AS worker

COPY CustomBoostTrack/ ./CustomBoostTrack/
RUN mkdir -p /app/models
COPY yolov8n.pt /app/models/yolov8n.pt

ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH=/app/.venv/bin:$PATH

RUN --mount=type=cache,target=/root/.cache/uv \
    python -m pip install --upgrade pip setuptools wheel uv && \
    uv sync --frozen --extra ml

ARG DOWNLOAD_MODELS=0
RUN --mount=type=cache,target=/root/.cache/mot-weights \
    if [ "${DOWNLOAD_MODELS}" = "1" ]; then \
        echo "Downloading model weights via scripts/download_models.py..."; \
        python scripts/download_models.py --output-dir /root/.cache/mot-weights --models yolo11 yolo12 osnet; \
        mkdir -p /app/models; \
        cp -a /root/.cache/mot-weights/. /app/models/; \
    else \
        echo "Skipping scripts/download_models.py (DOWNLOAD_MODELS=${DOWNLOAD_MODELS})"; \
    fi

RUN mkdir -p /app/var/uploads /app/var/results && \
    chown -R appuser:appuser /app

USER appuser

ENV HOST=0.0.0.0 \
    PORT=5000 \
    ENV=prod \
    PROJECT_ROOT=/app \
    UPLOAD_DIR=/app/var/uploads \
    RESULTS_DIR=/app/var/results \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MODELS_DIR=/app/models \
    YOLO11_MODEL_PATH=/app/models/yolo11x.pt \
    YOLO12_MODEL_PATH=/app/models/yolo12x.pt \
    REID_MODEL_PATH=/app/models/osnet_ain_ms_m_c.pth.tar \
    MOT_PIPELINE=boosttrack

CMD ["python", "-m", "mot_web.worker"]


# -----------------
# Worker target (ML, CUDA runtime base)
#
# This is intended for GPU machines. Keep it separate so laptops/CPU-only
# environments can still build/run the default `worker` target.
# -----------------
FROM ${CUDA_BASE_IMAGE} AS worker-cuda

LABEL maintainer="MOT Team"
LABEL description="Multi-Object Tracking Worker (CUDA runtime base)"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

ARG PIP_TIMEOUT=120
ARG PIP_RETRIES=10

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python-is-python3 \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

COPY pyproject.toml ./
COPY uv.lock ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY CustomBoostTrack/ ./CustomBoostTrack/

RUN mkdir -p /app/models
COPY yolov8n.pt /app/models/yolov8n.pt

# Optional: set TORCH_INDEX_URL to help pip select CUDA wheels on GPU builds.
# Example (CUDA 12.4): https://download.pytorch.org/whl/cu124
ARG TORCH_INDEX_URL=

ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH=/app/.venv/bin:$PATH

RUN --mount=type=cache,target=/root/.cache/uv \
    python -m pip install --upgrade pip setuptools wheel uv && \
    if [ -n "${TORCH_INDEX_URL}" ]; then export PIP_EXTRA_INDEX_URL="${TORCH_INDEX_URL}"; fi && \
    uv sync --frozen --extra ml

ARG DOWNLOAD_MODELS=0
RUN --mount=type=cache,target=/root/.cache/mot-weights \
    if [ "${DOWNLOAD_MODELS}" = "1" ]; then \
        echo "Downloading model weights via scripts/download_models.py..."; \
        python scripts/download_models.py --output-dir /root/.cache/mot-weights --models yolo11 yolo12 osnet; \
        mkdir -p /app/models; \
        cp -a /root/.cache/mot-weights/. /app/models/; \
    else \
        echo "Skipping scripts/download_models.py (DOWNLOAD_MODELS=${DOWNLOAD_MODELS})"; \
    fi

RUN mkdir -p /app/var/uploads /app/var/results && \
    chown -R appuser:appuser /app

USER appuser

ENV HOST=0.0.0.0 \
    PORT=5000 \
    ENV=prod \
    PROJECT_ROOT=/app \
    UPLOAD_DIR=/app/var/uploads \
    RESULTS_DIR=/app/var/results \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MODELS_DIR=/app/models \
    YOLO11_MODEL_PATH=/app/models/yolo11x.pt \
    YOLO12_MODEL_PATH=/app/models/yolo12x.pt \
    REID_MODEL_PATH=/app/models/osnet_ain_ms_m_c.pth.tar \
    MOT_PIPELINE=boosttrack

CMD ["python", "-m", "mot_web.worker"]


# -----------------
# Tests target
# -----------------
FROM base AS tests

COPY tests/ ./tests/
COPY pytest.ini ./pytest.ini
COPY pyrightconfig.json ./pyrightconfig.json

ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH=/app/.venv/bin:$PATH

RUN --mount=type=cache,target=/root/.cache/uv \
    python -m pip install --upgrade pip setuptools wheel uv && \
    uv sync --frozen --extra dev

RUN mkdir -p /app/var/uploads /app/var/results && \
    chown -R appuser:appuser /app

USER appuser

ENV ENV=test \
    PROJECT_ROOT=/app \
    UPLOAD_DIR=/app/var/uploads \
    RESULTS_DIR=/app/var/results \
    QUEUE_MODE=inline

CMD ["pytest", "-q", "tests"]


# -----------------
# Triton integration tests target (dev + ml)
# -----------------
FROM base AS itest-ml

COPY tests/ ./tests/
COPY pytest.ini ./pytest.ini

ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH=/app/.venv/bin:$PATH

RUN --mount=type=cache,target=/root/.cache/uv \
    python -m pip install --upgrade pip setuptools wheel uv && \
    uv sync --frozen --extra dev --extra ml

USER appuser

ENV ENV=test \
    PROJECT_ROOT=/app \
    UPLOAD_DIR=/app/var/uploads \
    RESULTS_DIR=/app/var/results

CMD ["pytest", "-q", "tests/test_triton_integration.py"]
