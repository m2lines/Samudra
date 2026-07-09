#!/bin/bash

# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

# Slurm + Apptainer harness for single-node evaluation, modeled after
# `skypilot/eval.sky.yaml` and `scripts/slurm_apptainer_train.sbatch`.
#
# Submit with env vars (example):
#   export CONFIG=configs/samudra_om4/eval.yaml
#   export NAME_SUFFIX=om4_samudra_baseline_eval  # NAME becomes YYYY-MM-DD-<suffix>
#   export TARGET_CHECKPOINT=2026-02-22-om4_samudra_baseline/saved_nets/ema_ckpt.pt
#   export ARGS="--num_model_steps_forward=25"
#   sbatch --gres=gpu:rtx6000:1 scripts/slurm_apptainer_eval.sbatch
#
# Container selection (pick one):
#   export IMAGE_REF=ghcr.io/m2lines/ocean-emulator-physicsnemo:25.11-<hash>
#   export CONTAINER_HASH=<git sha>   # expands to 25.11-<sha>
#   export CONTAINER_TAG=25.11-latest
#
# Checkpoint selection (pick one):
#   export TARGET_CHECKPOINT=<path-relative-to-OUTPUT_BASE>
#   export CKPT_PATH=/absolute/path/to/ema_ckpt.pt
#
# Private GHCR pulls (if needed):
#   export GHCR_USERNAME=...
#   export GHCR_TOKEN=...

#SBATCH --job-name=oe-eval
#SBATCH --output=slurm-%j.out
#SBATCH --account=torch_pr_347_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=04:00:00

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
BACKEND="${BACKEND:-cuda}"
if [[ -z "${NAME}" && -n "${NAME_SUFFIX}" ]]; then
  NAME="$(date +%Y-%m-%d)-${NAME_SUFFIX}"
fi
if [[ -z "${CONFIG}" || -z "${NAME}" ]]; then
  echo "Missing required env vars. Need CONFIG and NAME." >&2
  echo "Tip: set NAME_SUFFIX to auto-generate NAME as YYYY-MM-DD-<suffix>." >&2
  echo "Example: export NAME_SUFFIX=om4_samudra_baseline_eval" >&2
  exit 2
fi

# Config path inside the container image. Resolve relative paths under /workspace/
# (the image's working directory where configs/ is copied).
CONFIG_IN_CONTAINER="${CONFIG}"
if [[ "${CONFIG_IN_CONTAINER}" != /* ]]; then
  CONFIG_IN_CONTAINER="/workspace/${CONFIG_IN_CONTAINER}"
fi

# Data + outputs (host paths; will be bind-mounted into the container at same paths).
CURRENT_USER="${USER:-$(id -un)}"
DATA_ROOT_DEFAULT="/scratch/${CURRENT_USER}/data/om4_onedeg_v3"
OUTPUT_BASE_DEFAULT="/scratch/${CURRENT_USER}/runs"
DATA_ROOT="${DATA_ROOT:-${DATA_ROOT_DEFAULT}}"
OUTPUT_BASE="${OUTPUT_BASE:-${OUTPUT_BASE_DEFAULT}}"

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "DATA_ROOT directory does not exist: ${DATA_ROOT}" >&2
  echo "Set DATA_ROOT to an existing data directory." >&2
  echo "Example: export DATA_ROOT=${DATA_ROOT_DEFAULT}" >&2
  exit 3
fi
if [[ ! -d "${OUTPUT_BASE}" ]]; then
  echo "OUTPUT_BASE directory does not exist: ${OUTPUT_BASE}" >&2
  echo "Set OUTPUT_BASE to an existing output directory." >&2
  echo "Example: export OUTPUT_BASE=${OUTPUT_BASE_DEFAULT}" >&2
  exit 4
fi

RUN_DIR="${OUTPUT_BASE}/${NAME}"
if [[ -e "${RUN_DIR}" ]]; then
  echo "Refusing to run: output dir already exists: ${RUN_DIR}" >&2
  echo "Pick a unique NAME_SUFFIX." >&2
  exit 5
fi

# Resolve checkpoint path.
TARGET_CHECKPOINT="${TARGET_CHECKPOINT:-}"
CKPT_PATH="${CKPT_PATH:-}"
if [[ -n "${TARGET_CHECKPOINT}" && -n "${CKPT_PATH}" ]]; then
  echo "Specify only one of TARGET_CHECKPOINT or CKPT_PATH." >&2
  exit 6
fi
if [[ -n "${CKPT_PATH}" ]]; then
  if [[ "${CKPT_PATH}" = /* ]]; then
    CKPT_PATH_HOST="${CKPT_PATH}"
  else
    CKPT_PATH_HOST="$(realpath -m "${CKPT_PATH}")"
  fi
elif [[ -n "${TARGET_CHECKPOINT}" ]]; then
  if [[ "${TARGET_CHECKPOINT}" = /* ]]; then
    CKPT_PATH_HOST="${TARGET_CHECKPOINT}"
  else
    CKPT_PATH_HOST="${OUTPUT_BASE}/${TARGET_CHECKPOINT}"
  fi
else
  echo "Missing checkpoint. Set TARGET_CHECKPOINT (relative to OUTPUT_BASE) or CKPT_PATH." >&2
  exit 7
fi

if [[ ! -f "${CKPT_PATH_HOST}" ]]; then
  echo "Checkpoint not found: ${CKPT_PATH_HOST}" >&2
  exit 8
fi

# W&B mode defaults: online if key provided, else disabled.
WANDB_MODE="${WANDB_MODE:-}"
if [[ -z "${WANDB_MODE}" ]]; then
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    WANDB_MODE="online"
  else
    WANDB_MODE="disabled"
  fi
fi

# Container selection (same conventions as scripts/slurm_apptainer_train.sbatch).
CONTAINER_REGISTRY="${CONTAINER_REGISTRY:-ghcr.io}"
CONTAINER_ORG="${CONTAINER_ORG:-m2lines}"
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
# Default cache name keys off IMAGE_REF so distinct refs never collide.
IMAGE_REF_SAFE="$(echo "${IMAGE_REF}" | tr '/:@' '---')"
SIF_PATH="${SIF_PATH:-${SIF_DIR}/${IMAGE_REF_SAFE}.sif}"

# Apptainer tmp: keep on node-local /tmp to avoid scratch filesystems that don't support xattrs.
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/tmp}"

# GHCR auth (if image is private).
if [[ -n "${GHCR_USERNAME:-}" ]] && [[ -z "${APPTAINER_DOCKER_USERNAME:-}" ]]; then
  export APPTAINER_DOCKER_USERNAME="${GHCR_USERNAME}"
fi
if [[ -n "${GHCR_TOKEN:-}" ]] && [[ -z "${APPTAINER_DOCKER_PASSWORD:-}" ]]; then
  export APPTAINER_DOCKER_PASSWORD="${GHCR_TOKEN}"
fi

echo "Repo dir:   ${REPO_DIR}"
echo "Config:     ${CONFIG_IN_CONTAINER}"
echo "Name:       ${NAME}"
echo "Data root:  ${DATA_ROOT}"
echo "Outputs:    ${OUTPUT_BASE}"
echo "Checkpoint: ${CKPT_PATH_HOST}"
echo "W&B mode:   ${WANDB_MODE}"
echo "Backend:    ${BACKEND}"
echo "Image ref:  ${IMAGE_REF}"
echo "SIF path:   ${SIF_PATH}"

if [[ ! -s "${SIF_PATH}" ]]; then
  echo "SIF not found locally. Pulling from registry..."
  "${APPTAINER_BIN}" pull "${SIF_PATH}" "docker://${IMAGE_REF}"
else
  echo "Using existing SIF."
fi

if [[ -z "${OMP_NUM_THREADS:-}" ]]; then
  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
fi
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${OMP_NUM_THREADS}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${OMP_NUM_THREADS}}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export APPTAINERENV_PYTHONNOUSERSITE="${APPTAINERENV_PYTHONNOUSERSITE:-${PYTHONNOUSERSITE}}"
export SINGULARITYENV_PYTHONNOUSERSITE="${SINGULARITYENV_PYTHONNOUSERSITE:-${PYTHONNOUSERSITE}}"

# Bind mounts.
DATA_CACHE_DIR="${DATA_CACHE_DIR:-${REPO_DIR}/.data_cache/${NAME}}"
mkdir -p "${DATA_CACHE_DIR}"
binds=(
  "${DATA_ROOT}:${DATA_ROOT}"
  "${OUTPUT_BASE}:${OUTPUT_BASE}"
  "${DATA_CACHE_DIR}:/workspace/.data_cache"
)

# If checkpoint lives outside DATA_ROOT/OUTPUT_BASE, bind its parent path too.
case "${CKPT_PATH_HOST}" in
  "${OUTPUT_BASE}"|"${OUTPUT_BASE}"/*|"${DATA_ROOT}"|"${DATA_ROOT}"/*)
    ;;
  *)
    CKPT_DIR="$(dirname "${CKPT_PATH_HOST}")"
    binds+=("${CKPT_DIR}:${CKPT_DIR}")
    ;;
esac

BIND_ARG="$(IFS=,; echo "${binds[*]}")"

# Create the run directory after preflight validation so early failures are
# retryable with the same NAME.
mkdir -p "${RUN_DIR}"

srun --ntasks=1 --ntasks-per-node=1 \
  "${APPTAINER_BIN}" exec --nv \
    --bind "${BIND_ARG}" \
    --pwd /workspace \
    "${SIF_PATH}" \
    bash -lc "
      set -euo pipefail
      export PYTHONNOUSERSITE=1
      export PYTHONPATH=/workspace/src:\${PYTHONPATH:-}
      if [[ ! -x /workspace/.venv/bin/python ]]; then
        echo \"ERROR: /workspace/.venv/bin/python not found inside container\" >&2
        exit 1
      fi
      if [[ ! -f \"${CONFIG_IN_CONTAINER}\" ]]; then
        echo \"ERROR: config not found inside container: ${CONFIG_IN_CONTAINER}\" >&2
        echo \"Hint: rebuild/pull a container tag that includes /workspace/configs.\" >&2
        exit 2
      fi
      /workspace/.venv/bin/python -m samudra.eval \"${CONFIG_IN_CONTAINER}\" \
        --ckpt_path=\"${CKPT_PATH_HOST}\" \
        --experiment.name=\"${NAME}\" \
        --experiment.data_root=\"${DATA_ROOT}\" \
        --experiment.base_output_dir=\"${OUTPUT_BASE}\" \
        --experiment.wandb.mode=\"${WANDB_MODE}\" \
        --backend=\"${BACKEND}\" \
        ${ARGS}
    "
