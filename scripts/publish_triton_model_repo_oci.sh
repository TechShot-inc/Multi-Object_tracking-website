#!/usr/bin/env bash
set -euo pipefail

# Publishes a Triton "model repository" directory as a single OCI artifact layer.
# The artifact is meant to be pulled by docker/compose.yml's model-init service.
#
# Usage:
#   scripts/publish_triton_model_repo_oci.sh <oci-ref> [model-repo-dir]
#
# Example:
#   scripts/publish_triton_model_repo_oci.sh ghcr.io/your-org/mot-triton-models:yolo11-yolo12 ./models_repo
#   scripts/publish_triton_model_repo_oci.sh ghcr.io/your-org/mot-triton-models@sha256:... ./models_repo

REF="${1:-}"
MODEL_REPO_DIR="${2:-models_repo}"

if [[ -z "${REF}" ]]; then
  echo "ERROR: missing OCI ref"
  echo "Usage: $0 <oci-ref> [model-repo-dir]"
  exit 2
fi

if [[ ! -d "${MODEL_REPO_DIR}" ]]; then
  echo "ERROR: model repo dir not found: ${MODEL_REPO_DIR}"
  exit 2
fi

if ! command -v oras >/dev/null 2>&1; then
  echo "ERROR: oras CLI not found on PATH"
  echo "Install: https://oras.land/cli/"
  exit 2
fi

TMPDIR="$(mktemp -d -t triton-model-repo.XXXXXX)"
trap 'rm -rf "${TMPDIR}"' EXIT

# Use a stable, relative filename inside the artifact so `oras pull` can write it
# safely under its output directory (no path traversal).
ARCHIVE_NAME="triton-models.tar.gz"
TMP_TGZ="${TMPDIR}/${ARCHIVE_NAME}"

# Pack the directory so model-init can pull + extract in one shot.
# We standardize the filename so model-init can detect it.
# (This file is the single artifact layer.)
#
# Note: tarball content root is "." so extraction directly into /models works.
tar -C "${MODEL_REPO_DIR}" -czf "${TMP_TGZ}" .

echo "Pushing Triton model repo artifact: ${REF}"
PLAIN_HTTP=""
if [[ "${ORAS_PLAIN_HTTP:-0}" == "1" ]]; then
  PLAIN_HTTP="--plain-http"
fi

(cd "${TMPDIR}" && oras push \
  ${PLAIN_HTTP} \
  --artifact-type application/vnd.mot.triton.modelrepo.v1+tgz \
  "${REF}" \
  "${ARCHIVE_NAME}":application/vnd.oci.image.layer.v1.tar+gzip)

echo "Done. Configure deployment with:"
echo "  TRITON_MODEL_REPO_REF=${REF}"
