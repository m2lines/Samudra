#!/bin/bash

# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

# Slurm + Apptainer harness for the PCGB diagnostic — a single-GPU,
# read-only scan that scores every skip-mask on val and reports whether
# PCGB has slack to redistribute. Gates the full 12 GPU-hour training run.
#
# Submit with env vars (example):
#   export CONFIG=configs/samudra_om4_v2/boosted_pcgb_e1.yaml
#   export NAME_SUFFIX=pcgb_diagnostic_e1
#   export CKPT_PATH=/scratch/am16581/runs/om4_samudra_v2_dense_dilated_v1/saved_nets/ckpt.pt
#   export DATA_ROOT=/scratch/am16581/data/om4_onedeg_v3
#   sbatch scripts/slurm_apptainer_pcgb_diagnostic.sbatch
#
# Container selection — same conventions as scripts/slurm_apptainer_eval.sbatch.

#SBATCH --job-name=oe-pcgb-diag
#SBATCH --output=slurm-%j.out
#SBATCH --account=torch_pr_347_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
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

CONFIG="${CONFIG:-configs/samudra_om4_v2/boosted_pcgb_e1.yaml}"
NAME="${NAME:-}"
NAME_SUFFIX="${NAME_SUFFIX:-pcgb_diagnostic}"
ARGS="${ARGS:-}"
BACKEND="${BACKEND:-cuda}"
if [[ -z "${NAME}" && -n "${NAME_SUFFIX}" ]]; then
  NAME="$(date +%Y-%m-%d)-${NAME_SUFFIX}"
fi

CONFIG_IN_CONTAINER="${CONFIG}"
if [[ "${CONFIG_IN_CONTAINER}" != /* ]]; then
  CONFIG_IN_CONTAINER="/workspace/${CONFIG_IN_CONTAINER}"
fi

CURRENT_USER="${USER:-$(id -un)}"
DATA_ROOT_DEFAULT="/scratch/${CURRENT_USER}/data/om4_onedeg_v3"
OUTPUT_BASE_DEFAULT="/scratch/${CURRENT_USER}/runs"
DATA_ROOT="${DATA_ROOT:-${DATA_ROOT_DEFAULT}}"
OUTPUT_BASE="${OUTPUT_BASE:-${OUTPUT_BASE_DEFAULT}}"

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "DATA_ROOT directory does not exist: ${DATA_ROOT}" >&2
  exit 3
fi
if [[ ! -d "${OUTPUT_BASE}" ]]; then
  echo "OUTPUT_BASE directory does not exist: ${OUTPUT_BASE}" >&2
  exit 4
fi

RUN_DIR="${OUTPUT_BASE}/${NAME}"
if [[ -e "${RUN_DIR}" ]]; then
  echo "Refusing to run: output dir already exists: ${RUN_DIR}" >&2
  echo "Pick a unique NAME_SUFFIX." >&2
  exit 5
fi

# Checkpoint resolution — same UX as eval sbatch but using resume_ckpt_path
# (the diagnostic config's field name, inherited from TrainConfig).
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
  # Fall back to whatever resume_ckpt_path is set in the YAML.
  CKPT_PATH_HOST=""
fi
if [[ -n "${CKPT_PATH_HOST}" && ! -f "${CKPT_PATH_HOST}" ]]; then
  echo "Checkpoint not found: ${CKPT_PATH_HOST}" >&2
  exit 8
fi

WANDB_MODE="${WANDB_MODE:-}"
if [[ -z "${WANDB_MODE}" ]]; then
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    WANDB_MODE="online"
  else
    WANDB_MODE="disabled"
  fi
fi

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

echo "Repo dir:   ${REPO_DIR}"
echo "Config:     ${CONFIG_IN_CONTAINER}"
echo "Name:       ${NAME}"
echo "Data root:  ${DATA_ROOT}"
echo "Outputs:    ${OUTPUT_BASE}"
echo "Checkpoint: ${CKPT_PATH_HOST:-<from yaml>}"
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

# Bind-mount host repo source so PCGB changes are visible without a
# container rebuild — same convention as the kernel-branch train sbatch.
# Includes scripts/ so /workspace/scripts/pcgb_diagnostic.py resolves.
DATA_CACHE_DIR="${DATA_CACHE_DIR:-${REPO_DIR}/.data_cache/${NAME}}"
mkdir -p "${DATA_CACHE_DIR}"
binds=(
  "${DATA_ROOT}:${DATA_ROOT}"
  "${OUTPUT_BASE}:${OUTPUT_BASE}"
  "${DATA_CACHE_DIR}:/workspace/.data_cache"
)
if [[ "${MOUNT_SOURCE:-1}" != "0" ]]; then
  binds+=("${REPO_DIR}/src:/workspace/src")
  binds+=("${REPO_DIR}/configs:/workspace/configs")
  binds+=("${REPO_DIR}/scripts:/workspace/scripts")
fi

# Bind the parent dir of the checkpoint if it lives outside the
# already-bound roots, so the container can read it.
if [[ -n "${CKPT_PATH_HOST}" ]]; then
  case "${CKPT_PATH_HOST}" in
    "${OUTPUT_BASE}"|"${OUTPUT_BASE}"/*|"${DATA_ROOT}"|"${DATA_ROOT}"/*)
      ;;
    *)
      CKPT_DIR="$(dirname "${CKPT_PATH_HOST}")"
      binds+=("${CKPT_DIR}:${CKPT_DIR}")
      ;;
  esac
fi

BIND_ARG="$(IFS=,; echo "${binds[*]}")"
mkdir -p "${RUN_DIR}"

CKPT_OVERRIDE_ARG=""
if [[ -n "${CKPT_PATH_HOST}" ]]; then
  CKPT_OVERRIDE_ARG="--resume_ckpt_path=${CKPT_PATH_HOST}"
fi

srun --ntasks=1 --ntasks-per-node=1 \
  "${APPTAINER_BIN}" exec --nv \
    --bind "${BIND_ARG}" \
    --pwd /workspace \
    "${SIF_PATH}" \
    bash -lc "
      set -euo pipefail
      export PYTHONPATH=/workspace/src:\${PYTHONPATH:-}
      if [[ ! -x /workspace/.venv/bin/python ]]; then
        echo \"ERROR: /workspace/.venv/bin/python not found inside container\" >&2
        exit 1
      fi
      if [[ ! -f \"${CONFIG_IN_CONTAINER}\" ]]; then
        echo \"ERROR: config not found inside container: ${CONFIG_IN_CONTAINER}\" >&2
        exit 2
      fi
      if [[ ! -f /workspace/scripts/pcgb_diagnostic.py ]]; then
        echo \"ERROR: /workspace/scripts/pcgb_diagnostic.py not found\" >&2
        echo \"Hint: ensure MOUNT_SOURCE=1 (default) so the host scripts/ dir is bound.\" >&2
        exit 3
      fi
      /workspace/.venv/bin/python /workspace/scripts/pcgb_diagnostic.py \
        \"${CONFIG_IN_CONTAINER}\" \
        --experiment.name=\"${NAME}\" \
        --experiment.data_root=\"${DATA_ROOT}\" \
        --experiment.base_output_dir=\"${OUTPUT_BASE}\" \
        --experiment.wandb.mode=\"${WANDB_MODE}\" \
        --backend=\"${BACKEND}\" \
        ${CKPT_OVERRIDE_ARG} \
        ${ARGS}
    "
