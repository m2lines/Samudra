#!/bin/bash
# Slurm + Apptainer harness for memory profiling a single training rank.
#
# Runs memray (already installed in the container) on a single GPU without DDP
# to measure per-rank memory footprint.  No container rebuild required.
#
# Submit via scripts/launch_kr1_profile.sh or directly:
#   export CONFIG=configs/fomo_om4/train_multiscale.yaml
#   export NAME_SUFFIX=profile_test
#   export CONTAINER_HASH=<sha>
#   sbatch --gres=gpu:rtx6000:1 --mem=256G scripts/slurm_apptainer_profile.sbatch

#SBATCH --job-name=oe-profile
#SBATCH --output=slurm-%j.out
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=02:00:00

set -euo pipefail

if command -v apptainer >/dev/null 2>&1; then
  APPTAINER_BIN=apptainer
elif command -v singularity >/dev/null 2>&1; then
  APPTAINER_BIN=singularity
else
  echo "Neither apptainer nor singularity is available on PATH." >&2
  exit 1
fi

CONFIG="${CONFIG:-}"
NAME="${NAME:-}"
NAME_SUFFIX="${NAME_SUFFIX:-}"
ARGS="${ARGS:-}"
if [[ -z "${NAME}" && -n "${NAME_SUFFIX}" ]]; then
  NAME="$(date +%Y-%m-%d)-${NAME_SUFFIX}"
fi
if [[ -z "${CONFIG}" || -z "${NAME}" ]]; then
  echo "Missing required env vars. Need CONFIG and NAME (or NAME_SUFFIX)." >&2
  exit 2
fi

CONFIG_IN_CONTAINER="${CONFIG}"
if [[ "${CONFIG_IN_CONTAINER}" != /* ]]; then
  CONFIG_IN_CONTAINER="/workspace/${CONFIG_IN_CONTAINER}"
fi

CURRENT_USER="${USER:-$(id -un)}"
DATA_ROOT="${DATA_ROOT:-/scratch/${CURRENT_USER}/data/om4_onedeg_v3}"
OUTPUT_BASE="${OUTPUT_BASE:-/scratch/${CURRENT_USER}/runs}"
RUN_DIR="${OUTPUT_BASE}/${NAME}"
PROFILE_OUT="${PROFILE_OUT:-/scratch/${CURRENT_USER}/runs/memprofile.bin}"

# Container selection (same as train sbatch).
CONTAINER_REGISTRY="${CONTAINER_REGISTRY:-ghcr.io}"
CONTAINER_ORG="${CONTAINER_ORG:-open-athena}"
CONTAINER_REPO="${CONTAINER_REPO:-ocean-emulator-physicsnemo}"
CONTAINER_HASH="${CONTAINER_HASH:-}"
CONTAINER_TAG="${CONTAINER_TAG:-}"
IMAGE_REF="${IMAGE_REF:-}"
if [[ -z "${IMAGE_REF}" ]]; then
  if [[ -z "${CONTAINER_TAG}" ]]; then
    if [[ -n "${CONTAINER_HASH}" ]]; then
      CONTAINER_TAG="25.11-${CONTAINER_HASH}"
    else
      CONTAINER_TAG="25.11-latest"
    fi
  fi
  IMAGE_REF="${CONTAINER_REGISTRY}/${CONTAINER_ORG}/${CONTAINER_REPO}:${CONTAINER_TAG}"
fi

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_DIR}"

SIF_DIR="${SIF_DIR:-${REPO_DIR}/.apptainer-images}"
mkdir -p "${SIF_DIR}"
IMAGE_REF_SAFE="$(echo "${IMAGE_REF}" | tr '/:@' '---')"
SIF_PATH="${SIF_PATH:-${SIF_DIR}/${IMAGE_REF_SAFE}.sif}"

export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/tmp}"

if [[ -n "${GHCR_USERNAME:-}" ]] && [[ -z "${APPTAINER_DOCKER_USERNAME:-}" ]]; then
  export APPTAINER_DOCKER_USERNAME="${GHCR_USERNAME}"
fi
if [[ -n "${GHCR_TOKEN:-}" ]] && [[ -z "${APPTAINER_DOCKER_PASSWORD:-}" ]]; then
  export APPTAINER_DOCKER_PASSWORD="${GHCR_TOKEN}"
fi

echo "=== Memory Profiling (single GPU, no DDP) ==="
echo "Config:      ${CONFIG_IN_CONTAINER}"
echo "Name:        ${NAME}"
echo "Data root:   ${DATA_ROOT}"
echo "Profile out: ${PROFILE_OUT}"
echo "Image ref:   ${IMAGE_REF}"
echo "SIF path:    ${SIF_PATH}"
echo "ARGS:        ${ARGS}"

if [[ ! -s "${SIF_PATH}" ]]; then
  echo "SIF not found locally. Pulling from registry..."
  "${APPTAINER_BIN}" pull "${SIF_PATH}" "docker://${IMAGE_REF}"
else
  echo "Using existing SIF."
fi

# Bind mounts: data + output + profile output directory.
binds=(
  "${DATA_ROOT}:${DATA_ROOT}"
  "${OUTPUT_BASE}:${OUTPUT_BASE}"
)
BIND_ARG=$(IFS=,; echo "${binds[*]}")

mkdir -p "${RUN_DIR}" "$(dirname "${PROFILE_OUT}")"

srun --ntasks=1 \
  "${APPTAINER_BIN}" exec --nv \
    --bind "${BIND_ARG}" \
    --pwd /workspace \
    "${SIF_PATH}" \
    bash -lc "
      set -euo pipefail
      export PYTHONPATH=/workspace/src:\${PYTHONPATH:-}

      echo '--- Running memray profile ---'
      /workspace/.venv/bin/python -m memray run \
        -o '${PROFILE_OUT}' \
        -m ocean_emulators.train '${CONFIG_IN_CONTAINER}' \
        --experiment.name='${NAME}' \
        --experiment.data_root='${DATA_ROOT}' \
        --experiment.base_output_dir='${OUTPUT_BASE}' \
        --experiment.wandb.mode=disabled \
        --backend=nccl \
        ${ARGS}

      echo '--- Generating memray stats ---'
      /workspace/.venv/bin/python -m memray stats '${PROFILE_OUT}'

      echo '--- Generating flamegraph ---'
      /workspace/.venv/bin/python -m memray flamegraph \
        '${PROFILE_OUT}' \
        -o '${PROFILE_OUT%.bin}_flamegraph.html'

      echo 'Profile complete. Files:'
      ls -lh '${PROFILE_OUT}' '${PROFILE_OUT%.bin}_flamegraph.html'
    "
