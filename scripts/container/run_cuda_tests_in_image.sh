#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-ocean-emulator:physicsnemo-26.05}"
PYTEST_MARK_EXPR="${PYTEST_MARK_EXPR:-cuda and not manual}"
PYTEST_ARGS="${PYTEST_ARGS:-}"

if [[ -n "${DOCKER_CMD:-}" ]]; then
  read -r -a docker_cmd <<<"${DOCKER_CMD}"
  if ! "${docker_cmd[@]}" version >/dev/null 2>&1; then
    echo "DOCKER_CMD is set but not usable: ${DOCKER_CMD}" >&2
    exit 1
  fi
else
  docker_cmd=(docker)
  if ! docker version >/dev/null 2>&1; then
    if command -v sudo >/dev/null 2>&1 && sudo -n docker version >/dev/null 2>&1; then
      docker_cmd=(sudo -n docker)
    else
      echo "Docker is required but not available to the current user." >&2
      exit 1
    fi
  fi
fi

echo "==> Running CUDA tests in ${IMAGE_TAG}"
"${docker_cmd[@]}" run --rm \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$PWD":/repo \
  -w /workspace \
  "${IMAGE_TAG}" \
  bash -lc "
    . .venv/bin/activate
    cd /repo
    python -m pytest -p no:zarr -q -m \"${PYTEST_MARK_EXPR}\" ${PYTEST_ARGS}
  "
