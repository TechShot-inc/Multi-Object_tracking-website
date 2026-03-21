#!/usr/bin/env bash
set -euo pipefail

# Runs a GitHub Actions self-hosted runner for exactly one job.
# This is the safest default for personal machines (not 24/7).
#
# Usage:
#   RUNNER_DIR=/mnt/Extra/actions-runner ./scripts/run_self_hosted_runner_once.sh
#
# Notes:
# - The runner must already be configured (config.sh has been run once).
# - This exits automatically after it completes a single job.

RUNNER_DIR="${RUNNER_DIR:-}"

if [[ -z "${RUNNER_DIR}" ]]; then
  echo "RUNNER_DIR is required. Example:" >&2
  echo "  RUNNER_DIR=/mnt/Extra/actions-runner $0" >&2
  exit 2
fi

if [[ ! -d "${RUNNER_DIR}" ]]; then
  echo "Runner dir not found: ${RUNNER_DIR}" >&2
  exit 2
fi

if [[ ! -x "${RUNNER_DIR}/run.sh" ]]; then
  echo "Missing or non-executable: ${RUNNER_DIR}/run.sh" >&2
  exit 2
fi

if [[ ! -f "${RUNNER_DIR}/.runner" ]]; then
  echo "Runner does not look configured (missing ${RUNNER_DIR}/.runner)." >&2
  echo "Go to GitHub -> Settings -> Actions -> Runners -> New self-hosted runner and run config.sh first." >&2
  exit 2
fi

pushd "${RUNNER_DIR}" >/dev/null

echo "Starting self-hosted runner in --once mode (will exit after one job)..."
./run.sh --once

popd >/dev/null
