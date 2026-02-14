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

GPU=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU=1
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Usage: ./scripts/run_triton_stack.sh [--gpu]

Requires:
  TRITON_MODEL_REPO_REF  OCI ref for model-init to pull into /models
Optional:
  ORAS_PLAIN_HTTP=1      For local/insecure HTTP registries
EOF
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${TRITON_MODEL_REPO_REF:-}" ]]; then
  echo "TRITON_MODEL_REPO_REF is not set." >&2
  echo "Example: export TRITON_MODEL_REPO_REF='ghcr.io/<you>/<repo>@sha256:<digest>'" >&2
  exit 1
fi

compose_files=("-f" "docker/compose.yml")
if [[ "$GPU" == "1" ]]; then
  compose_files+=("-f" "docker/compose.gpu.yml")
fi

export DETECTOR_BACKEND=triton
export ORAS_PLAIN_HTTP="${ORAS_PLAIN_HTTP:-0}"

set -x

docker compose "${compose_files[@]}" --profile triton up -d --build

echo
echo "UI: http://localhost:5000"
