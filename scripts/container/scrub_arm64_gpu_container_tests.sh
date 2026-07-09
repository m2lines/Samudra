#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

# This script is run on our local arm64 GPU box nightly in order to smoke test our latest docker image.
set -euo pipefail

PYTEST_MARK_EXPR="${PYTEST_MARK_EXPR:-cuda and not manual}"
PYTEST_ARGS="${PYTEST_ARGS:--x}"
IMAGE_TAG="${IMAGE_TAG:-}"
DOCKER_REPO="${DOCKER_REPO:-ghcr.io/m2lines}"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "This scrub must run on an ARM64/aarch64 host; got $(uname -m)." >&2
  exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required for the ARM GPU container scrub." >&2
  exit 3
fi

if ! nvidia-smi >/dev/null; then
  echo "nvidia-smi failed; no usable NVIDIA GPU is visible." >&2
  exit 4
fi

docker_cmd=(docker)
if ! docker version >/dev/null 2>&1; then
  if command -v sudo >/dev/null 2>&1 && sudo -n docker version >/dev/null 2>&1; then
    docker_cmd=(sudo -n docker)
  else
    echo "Docker is required for the ARM GPU container scrub and must be usable by the current user." >&2
    exit 5
  fi
fi

if [[ -z "${IMAGE_TAG}" ]]; then
  docker_repo_lower="$(echo "${DOCKER_REPO}" | tr '[:upper:]' '[:lower:]')"
  IMAGE_TAG="${docker_repo_lower}/ocean-emulator-physicsnemo:25.11-arm64-latest"
fi

login_token="${GHCR_TOKEN:-${GITHUB_TOKEN:-${GH_TOKEN:-}}}"
login_user="${GHCR_USERNAME:-${GITHUB_ACTOR:-}}"
if [[ -n "${login_token}" && -z "${login_user}" ]] && command -v gh >/dev/null 2>&1; then
  login_user="$(gh api user --jq .login 2>/dev/null || true)"
fi
if [[ -n "${login_token}" && -n "${login_user}" ]]; then
  echo "==> Logging in to ghcr.io as ${login_user}"
  printf '%s' "${login_token}" | "${docker_cmd[@]}" login ghcr.io -u "${login_user}" --password-stdin >/dev/null
fi

nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

echo "==> Pulling ${IMAGE_TAG}"
"${docker_cmd[@]}" pull "${IMAGE_TAG}"

echo "==> Running CUDA tests from ${IMAGE_TAG}"
DOCKER_CMD="${docker_cmd[*]}" \
IMAGE_TAG="${IMAGE_TAG}" \
PYTEST_MARK_EXPR="${PYTEST_MARK_EXPR}" \
PYTEST_ARGS="${PYTEST_ARGS}" \
bash scripts/container/run_cuda_tests_in_image.sh
