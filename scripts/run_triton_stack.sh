#!/usr/bin/env bash
set -euo pipefail

# Bring up the app using Triton as detector backend.
#
# Examples:
#   export TRITON_MODEL_REPO_REF='ghcr.io/<you>/<repo>@sha256:<digest>'
#   ./scripts/run_triton_stack.sh
#
#   export TRITON_MODEL_REPO_REF='mot-registry:5000/mot-triton-models@sha256:<digest>'
#   export ORAS_PLAIN_HTTP=1
#   ./scripts/run_triton_stack.sh --gpu

#   # Host-mounted model repo (skips ORAS pull)
#   export TRITON_HOST_MODEL_REPO='/mnt/Extra/mot-triton-model-repo'
#   ./scripts/run_triton_stack.sh --gpu

GPU=0
MODEL_REPO_SOURCE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU=1
      shift
      ;;
    --model-repo-source)
      MODEL_REPO_SOURCE="${2:-}"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage: ./scripts/run_triton_stack.sh [--gpu] [--model-repo-source host|oci]

Model repo sources:
  oci   Uses model-init (ORAS) to pull an OCI artifact into /models.
  host  Mounts a host directory into /models (skips ORAS pull).

If --model-repo-source is not provided, the script auto-selects:
  - oci  if TRITON_MODEL_REPO_REF is set
  - host if TRITON_HOST_MODEL_REPO is set
  - oci  otherwise

Optional:
  ORAS_PLAIN_HTTP=1      For local/insecure HTTP registries

OCI mode requires:
  TRITON_MODEL_REPO_REF  OCI ref for model-init to pull into /models

Host mode uses:
  TRITON_HOST_MODEL_REPO Host path to a Triton model repo directory
  (or a local ./models_repo directory if present)
EOF
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$MODEL_REPO_SOURCE" ]]; then
  if [[ -n "${TRITON_MODEL_REPO_REF:-}" ]]; then
    MODEL_REPO_SOURCE="oci"
  elif [[ -n "${TRITON_HOST_MODEL_REPO:-}" ]]; then
    MODEL_REPO_SOURCE="host"
  else
    MODEL_REPO_SOURCE="oci"
  fi
fi

case "$MODEL_REPO_SOURCE" in
  oci)
    if [[ -z "${TRITON_MODEL_REPO_REF:-}" ]]; then
      echo "TRITON_MODEL_REPO_REF is not set (required for --model-repo-source oci)." >&2
      echo "Example: export TRITON_MODEL_REPO_REF='ghcr.io/<you>/<repo>@sha256:<digest>'" >&2
      echo "Alternative: set TRITON_HOST_MODEL_REPO and use --model-repo-source host" >&2
      exit 1
    fi
    ;;
  host)
    if [[ -z "${TRITON_HOST_MODEL_REPO:-}" ]]; then
      if [[ ! -d "models_repo" ]]; then
        echo "TRITON_HOST_MODEL_REPO is not set and ./models_repo does not exist." >&2
        echo "Example: export TRITON_HOST_MODEL_REPO='/mnt/Extra/mot-triton-model-repo'" >&2
        exit 1
      fi
    fi
    ;;
  *)
    echo "Invalid --model-repo-source: $MODEL_REPO_SOURCE (expected host|oci)" >&2
    exit 2
    ;;
esac

compose_files=("-f" "docker/compose.yml")
if [[ "$GPU" == "1" ]]; then
  compose_files+=("-f" "docker/compose.gpu.yml")
fi
if [[ "$MODEL_REPO_SOURCE" == "host" ]]; then
  compose_files+=("-f" "docker/compose.triton.localrepo.yml")
fi

export DETECTOR_BACKEND=triton
export ORAS_PLAIN_HTTP="${ORAS_PLAIN_HTTP:-0}"
export TRITON_HOST_MODEL_REPO="${TRITON_HOST_MODEL_REPO:-}"

set -x

docker compose "${compose_files[@]}" --profile triton up -d --build

echo
echo "UI: http://localhost:5000"
