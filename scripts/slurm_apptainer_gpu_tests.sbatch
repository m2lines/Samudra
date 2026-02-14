#!/bin/bash
#SBATCH --job-name=oe-apptainer-gpu-tests
#SBATCH --output=slurm-%j.out
#SBATCH --account=torch_pr_347_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=1000GB
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=02:00:00

set -euo pipefail

# Optional: set APPTAINER_MODULE if your cluster requires a module load.
if [[ -n "${APPTAINER_MODULE:-}" ]] && command -v module >/dev/null 2>&1; then
  module load "${APPTAINER_MODULE}"
fi

if command -v apptainer >/dev/null 2>&1; then
  APPTAINER_BIN=apptainer
elif command -v singularity >/dev/null 2>&1; then
  APPTAINER_BIN=singularity
else
  echo "Neither apptainer nor singularity is available on PATH." >&2
  exit 1
fi

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_DIR}"

# Container selection:
# 1) IMAGE_REF (full image ref) takes priority.
# 2) else CONTAINER_TAG.
# 3) else CONTAINER_HASH -> tag 25.11-<hash>.
# 4) else default 25.11-latest.
CONTAINER_REGISTRY="${CONTAINER_REGISTRY:-ghcr.io}"
CONTAINER_ORG="${CONTAINER_ORG:-open-athena}"
CONTAINER_REPO="${CONTAINER_REPO:-ocean-emulator-physicsnemo}"
CONTAINER_HASH="${CONTAINER_HASH:-}"
CONTAINER_TAG="${CONTAINER_TAG:-}"
if [[ -z "${CONTAINER_TAG}" ]]; then
  if [[ -n "${CONTAINER_HASH}" ]]; then
    CONTAINER_TAG="25.11-${CONTAINER_HASH}"
  else
    CONTAINER_TAG="25.11-latest"
  fi
fi
IMAGE_REF="${IMAGE_REF:-${CONTAINER_REGISTRY}/${CONTAINER_ORG}/${CONTAINER_REPO}:${CONTAINER_TAG}}"

# Pull location and cache.
SIF_DIR="${SIF_DIR:-${REPO_DIR}/.apptainer-images}"
mkdir -p "${SIF_DIR}"
TAG_SAFE="$(echo "${CONTAINER_TAG}" | tr '/:@' '---')"
SIF_PATH="${SIF_PATH:-${SIF_DIR}/${CONTAINER_REPO}-${TAG_SAFE}.sif}"

# GHCR auth (if image is private).
if [[ -n "${GHCR_USERNAME:-}" ]] && [[ -z "${APPTAINER_DOCKER_USERNAME:-}" ]]; then
  export APPTAINER_DOCKER_USERNAME="${GHCR_USERNAME}"
fi
if [[ -n "${GHCR_TOKEN:-}" ]] && [[ -z "${APPTAINER_DOCKER_PASSWORD:-}" ]]; then
  export APPTAINER_DOCKER_PASSWORD="${GHCR_TOKEN}"
fi
if [[ -n "${APPTAINER_CACHEDIR:-}" ]]; then
  mkdir -p "${APPTAINER_CACHEDIR}"
fi

echo "Image ref: ${IMAGE_REF}"
echo "SIF path:  ${SIF_PATH}"
echo "Repo dir:  ${REPO_DIR}"

if [[ ! -s "${SIF_PATH}" ]]; then
  echo "SIF not found locally. Pulling from GHCR..."
  "${APPTAINER_BIN}" pull "${SIF_PATH}" "docker://${IMAGE_REF}"
else
  echo "Using existing SIF."
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

DATA_CACHE_DIR="${DATA_CACHE_DIR:-${REPO_DIR}/.data_cache}"
PYTEST_MARK_EXPR="${PYTEST_MARK_EXPR:-not manual and cuda}"
PYTEST_WORKERS="${PYTEST_WORKERS:-2}"
PYTEST_ARGS="${PYTEST_ARGS:--x}"

binds=("${REPO_DIR}:/repo")
if [[ -d "${DATA_CACHE_DIR}" ]]; then
  binds+=("${DATA_CACHE_DIR}:/repo/.data_cache")
fi
BIND_ARG="$(IFS=,; echo "${binds[*]}")"

echo "Running CUDA tests with marker: ${PYTEST_MARK_EXPR}"
"${APPTAINER_BIN}" exec --nv \
  --bind "${BIND_ARG}" \
  --pwd /repo \
  "${SIF_PATH}" \
  bash -lc "
    set -euo pipefail
    . /workspace/.venv/bin/activate
    export PYTHONPATH=/repo/src:\${PYTHONPATH:-}
    python -m pytest -n ${PYTEST_WORKERS} -v -m \"${PYTEST_MARK_EXPR}\" ${PYTEST_ARGS}
  "
