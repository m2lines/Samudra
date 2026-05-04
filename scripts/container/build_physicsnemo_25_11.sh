#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-ocean-emulator:physicsnemo-25.11}"
DOCKERFILE="${DOCKERFILE:-containers/Dockerfile.physicsnemo-25.11}"
BUILD_APPTAINER="${BUILD_APPTAINER:-1}"
RUN_SMOKE_TEST="${RUN_SMOKE_TEST:-1}"
SIF_PATH="${SIF_PATH:-dist/ocean-emulator_physicsnemo-25.11.sif}"

docker_cmd=(docker)
if ! docker version >/dev/null 2>&1; then
  if command -v sudo >/dev/null 2>&1 && sudo docker version >/dev/null 2>&1; then
    docker_cmd=(sudo docker)
  else
    echo "Docker is required but not available to the current user." >&2
    exit 1
  fi
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "This script must be run from within a git checkout." >&2
  exit 2
fi

if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "Refusing to build from a dirty checkout." >&2
  echo "Please commit and push your changes first, then rerun this build." >&2
  echo "" >&2
  git status --short >&2
  exit 3
fi

GIT_SHA="$(git rev-parse --verify HEAD)"
if ! GIT_URL="$(git config --get remote.origin.url)"; then
  echo "Could not determine git remote URL (remote.origin.url is unset)." >&2
  echo "Please configure origin, commit, and push before building." >&2
  exit 4
fi
BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "==> Building ${IMAGE_TAG} from ${DOCKERFILE}"
DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}" "${docker_cmd[@]}" build --pull \
  --build-arg VCS_REF="${GIT_SHA}" \
  --build-arg VCS_URL="${GIT_URL}" \
  --build-arg BUILD_DATE="${BUILD_DATE}" \
  -f "${DOCKERFILE}" \
  -t "${IMAGE_TAG}" \
  .

if [[ "${RUN_SMOKE_TEST}" != "0" ]]; then
  echo "==> Running smoke test in ${IMAGE_TAG}"
  "${docker_cmd[@]}" run --rm --entrypoint bash "${IMAGE_TAG}" -lc \
    ". .venv/bin/activate && python scripts/container/smoke_test.py"
else
  echo "==> Skipping smoke test (RUN_SMOKE_TEST=0)"
fi

if [[ "${BUILD_APPTAINER}" == "0" ]]; then
  echo "==> Skipping SIF build (BUILD_APPTAINER=0)"
  exit 0
fi

apptainer_cmd=()
if command -v apptainer >/dev/null 2>&1; then
  apptainer_cmd=(apptainer)
elif command -v singularity >/dev/null 2>&1; then
  apptainer_cmd=(singularity)
fi

if [[ "${#apptainer_cmd[@]}" -eq 0 ]]; then
  echo "==> apptainer/singularity not found; skipping SIF build"
  exit 0
fi

echo "==> Building SIF ${SIF_PATH} from ${IMAGE_TAG}"
mkdir -p "$(dirname "${SIF_PATH}")"
"${apptainer_cmd[@]}" build --force "${SIF_PATH}" "docker-daemon://${IMAGE_TAG}"
