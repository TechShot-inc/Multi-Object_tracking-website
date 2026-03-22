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

# Optional env vars:
#   OUT_DIR            Directory to write bundle artifacts to (default: var/model_bundle)
#   ORAS_DRY_RUN=1     Build bundle artifacts but do not push
#   ORAS_PLAIN_HTTP=1  Use plain-http (for insecure registries)
#   ORAS_CONTAINER_IMAGE=ghcr.io/oras-project/oras:v1.2.0

if [[ -z "${REF}" ]]; then
  echo "ERROR: missing OCI ref"
  echo "Usage: $0 <oci-ref> [model-repo-dir]"
  exit 2
fi

if [[ "${REF}" == *"@sha256:"* ]]; then
  echo "ERROR: OCI ref must be a tag when pushing (got a digest ref): ${REF}" >&2
  echo "Hint: use something like ghcr.io/<org>/<name>:<version>; the digest is produced by the push." >&2
  exit 2
fi

if [[ ! -d "${MODEL_REPO_DIR}" ]]; then
  echo "ERROR: model repo dir not found: ${MODEL_REPO_DIR}"
  exit 2
fi

OUT_DIR="${OUT_DIR:-var/model_bundle}"
mkdir -p "${OUT_DIR}"

ORAS_IMAGE="${ORAS_CONTAINER_IMAGE:-ghcr.io/oras-project/oras:v1.2.0}"

ORAS_BIN="oras"
if ! command -v oras >/dev/null 2>&1; then
  if command -v docker >/dev/null 2>&1; then
    ORAS_BIN="docker-run-oras"
  else
    echo "ERROR: oras CLI not found on PATH and docker is not available for fallback" >&2
    echo "Install ORAS: https://oras.land/cli/" >&2
    exit 2
  fi
fi

TMPDIR="$(mktemp -d -t triton-model-repo.XXXXXX)"
trap 'rm -rf "${TMPDIR}"' EXIT

# Use a stable, relative filename inside the artifact so `oras pull` can write it
# safely under its output directory (no path traversal).
ARCHIVE_NAME="triton-models.tar.gz"
TMP_TGZ="${TMPDIR}/${ARCHIVE_NAME}"

TMP_REPO_DIR="${TMPDIR}/repo"
mkdir -p "${TMP_REPO_DIR}"

# Copy into a staging dir so we can add manifest files without mutating the source.
cp -a "${MODEL_REPO_DIR}/." "${TMP_REPO_DIR}/"

# Generate deterministic manifest + checksums for the repo.
python3 scripts/manifest_dir.py \
  --dir "${TMP_REPO_DIR}" \
  --out-json "${TMPDIR}/model_repo_manifest.json" \
  --out-sha256 "${TMPDIR}/model_repo.SHA256SUMS" \
  --root-prefix "models_repo"

cp -a "${TMPDIR}/model_repo_manifest.json" "${TMP_REPO_DIR}/model_repo_manifest.json"
cp -a "${TMPDIR}/model_repo.SHA256SUMS" "${TMP_REPO_DIR}/model_repo.SHA256SUMS"

# Pack the directory so model-init can pull + extract in one shot.
# We standardize the filename so model-init can detect it.
# (This file is the single artifact layer.)
#
# Note: tarball content root is "." so extraction directly into /models works.
tar -C "${TMP_REPO_DIR}" -czf "${TMP_TGZ}" .

TAR_SHA256="$(sha256sum "${TMP_TGZ}" | awk '{print $1}')"
echo "${TAR_SHA256}  ${ARCHIVE_NAME}" > "${TMPDIR}/${ARCHIVE_NAME}.sha256"

cp -a "${TMP_TGZ}" "${OUT_DIR}/${ARCHIVE_NAME}"
cp -a "${TMPDIR}/${ARCHIVE_NAME}.sha256" "${OUT_DIR}/${ARCHIVE_NAME}.sha256"
cp -a "${TMPDIR}/model_repo_manifest.json" "${OUT_DIR}/model_repo_manifest.json"
cp -a "${TMPDIR}/model_repo.SHA256SUMS" "${OUT_DIR}/model_repo.SHA256SUMS"

echo "Pushing Triton model repo artifact: ${REF}"
PLAIN_HTTP=""
if [[ "${ORAS_PLAIN_HTTP:-0}" == "1" ]]; then
  PLAIN_HTTP="--plain-http"
fi

if [[ "${ORAS_DRY_RUN:-0}" == "1" ]]; then
  echo "ORAS_DRY_RUN=1 set; skipping push. Bundle artifacts are in: ${OUT_DIR}"
  echo "Would push ref: ${REF}"
  exit 0
fi

oras_push() {
  if [[ "${ORAS_BIN}" == "docker-run-oras" ]]; then
    DOCKER_AUTH_MOUNT=()
    if [[ -d "${HOME:-}/.docker" ]]; then
      # Reuse `docker login` credentials (GitHub Actions writes to $HOME/.docker/config.json).
      DOCKER_AUTH_MOUNT=( -v "${HOME}/.docker:/root/.docker:ro" )
    fi
    docker run --rm \
      -v "${TMPDIR}:/work" \
      -w /work \
      "${DOCKER_AUTH_MOUNT[@]}" \
      "${ORAS_IMAGE}" push \
      ${PLAIN_HTTP} \
      --artifact-type application/vnd.mot.triton.modelrepo.v1+tgz \
      "${REF}" \
      "${ARCHIVE_NAME}":application/vnd.oci.image.layer.v1.tar+gzip
  else
    (cd "${TMPDIR}" && oras push \
      ${PLAIN_HTTP} \
      --artifact-type application/vnd.mot.triton.modelrepo.v1+tgz \
      "${REF}" \
      "${ARCHIVE_NAME}":application/vnd.oci.image.layer.v1.tar+gzip)
  fi
}

oras_push

echo "Done. Configure deployment with:"
echo "  TRITON_MODEL_REPO_REF=${REF}"
echo
echo "Bundle checksums/manifest written to: ${OUT_DIR}"
