#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-ocean-emulator:physicsnemo-25.11}"
DOCKERFILE="${DOCKERFILE:-containers/Dockerfile.physicsnemo-25.11}"
BUILD_APPTAINER="${BUILD_APPTAINER:-1}"
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

echo "==> Building ${IMAGE_TAG} from ${DOCKERFILE}"
"${docker_cmd[@]}" build --pull -f "${DOCKERFILE}" -t "${IMAGE_TAG}" .

echo "==> Running smoke test in ${IMAGE_TAG}"
"${docker_cmd[@]}" run --rm --entrypoint bash "${IMAGE_TAG}" -lc \
  ". .venv/bin/activate && python scripts/container/smoke_test.py"

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
