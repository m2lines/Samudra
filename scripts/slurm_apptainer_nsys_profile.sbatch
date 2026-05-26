#!/bin/bash

# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

# Slurm + Apptainer harness for profiling distributed training with NVIDIA Nsight Systems.
#
# This is a modified version of slurm_apptainer_train.sbatch that wraps
# torchrun with `nsys profile`.  It accepts the same env vars as the train
# harness (CONFIG, NAME, DATA_ROOT, …) plus:
#
#   NSYS_CMD  — the full `nsys profile …` prefix (built by the launcher script)
#
# The resulting .nsys-rep files are written into the run directory and can be
# opened in the Nsight Systems GUI.
#
# Do NOT ctrl-C the job — let nsys stop on its own (via --duration) or let the
# training process exit.  ctrl-C causes nsys to lose data.

#SBATCH --job-name=oe-nsys
#SBATCH --output=slurm-%j.out
#SBATCH --account=torch_pr_347_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=1400G
#SBATCH --gres=gpu:rtx6000:8
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

# Data + outputs.
CURRENT_USER="${USER:-$(id -un)}"
DATA_ROOT_DEFAULT="/scratch/${CURRENT_USER}/data/om4_onedeg_v3"
OUTPUT_BASE_DEFAULT="/scratch/${CURRENT_USER}/runs"
DATA_ROOT="${DATA_ROOT:-${DATA_ROOT_DEFAULT}}"
OUTPUT_BASE="${OUTPUT_BASE:-${OUTPUT_BASE_DEFAULT}}"

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "DATA_ROOT directory does not exist: ${DATA_ROOT}" >&2
  exit 5
fi
if [[ ! -d "${OUTPUT_BASE}" ]]; then
  echo "OUTPUT_BASE directory does not exist: ${OUTPUT_BASE}" >&2
  exit 6
fi

RUN_DIR="${OUTPUT_BASE}/${NAME}"
if [[ -e "${RUN_DIR}" ]]; then
  echo "Refusing to run: output dir already exists: ${RUN_DIR}" >&2
  echo "Pick a unique NAME_SUFFIX." >&2
  exit 4
fi

# W&B defaults: disabled for profiling unless explicitly set.
WANDB_MODE="${WANDB_MODE:-disabled}"

# Container selection.
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
echo "W&B mode:   ${WANDB_MODE}"
echo "Image ref:  ${IMAGE_REF}"
echo "SIF path:   ${SIF_PATH}"
echo "NSYS_CMD:   ${NSYS_CMD:-<not set>}"

if [[ ! -s "${SIF_PATH}" ]]; then
  echo "SIF not found locally. Pulling from registry..."
  "${APPTAINER_BIN}" pull "${SIF_PATH}" "docker://${IMAGE_REF}"
else
  echo "Using existing SIF."
fi

# Distributed rendezvous.
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-8008}"
export MASTER_ADDR MASTER_PORT

GPUS_PER_NODE="${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-}}"
if [[ -z "${GPUS_PER_NODE}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS_PER_NODE="$(nvidia-smi -L | wc -l | tr -d ' ')"
  else
    echo "Could not determine GPUS_PER_NODE." >&2
    exit 3
  fi
fi

if [[ -z "${OMP_NUM_THREADS:-}" ]]; then
  if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
    t="$(( SLURM_CPUS_PER_TASK / GPUS_PER_NODE ))"
    if (( t < 1 )); then t=1; fi
    export OMP_NUM_THREADS="$t"
  else
    export OMP_NUM_THREADS=1
  fi
fi
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${OMP_NUM_THREADS}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${OMP_NUM_THREADS}}"
export PYTHONUNBUFFERED=1

# Bind mounts.
DATA_CACHE_DIR="${DATA_CACHE_DIR:-${REPO_DIR}/.data_cache/${NAME}}"
mkdir -p "${DATA_CACHE_DIR}"
binds=(
  "${DATA_ROOT}:${DATA_ROOT}"
  "${OUTPUT_BASE}:${OUTPUT_BASE}"
  "${DATA_CACHE_DIR}:/workspace/.data_cache"
)
BIND_ARG="$(IFS=,; echo "${binds[*]}")"

mkdir -p "${RUN_DIR}"

# ── nsys profile command ──
# If NSYS_CMD is not set, build a reasonable default.
if [[ -z "${NSYS_CMD:-}" ]]; then
  NSYS_CMD="nsys profile --trace=cuda,nvtx,osrt,nccl --sample=cpu --duration=600 --delay=120 --kill=sigterm --output=${RUN_DIR}/nsys_%h_%p --force-overwrite=true --stats=true"
fi

srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
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
        echo \"ERROR: config not found: ${CONFIG_IN_CONTAINER}\" >&2
        exit 2
      fi

      # Wrap torchrun with nsys.  nsys will profile the torchrun parent and
      # all forked worker processes, producing one .nsys-rep per process.
      ${NSYS_CMD} \
        /workspace/.venv/bin/python -m torch.distributed.run \
          --nnodes=${SLURM_NNODES} \
          --nproc_per_node=${GPUS_PER_NODE} \
          --master_addr=${MASTER_ADDR} \
          --master_port=${MASTER_PORT} \
          --node_rank=\${SLURM_PROCID} \
          -m ocean_emulators.train \"${CONFIG_IN_CONTAINER}\" \
          --experiment.name=\"${NAME}\" \
          --experiment.data_root=\"${DATA_ROOT}\" \
          --experiment.base_output_dir=\"${OUTPUT_BASE}\" \
          --experiment.wandb.mode=\"${WANDB_MODE}\" \
          --backend=\"nccl\" \
          ${ARGS}
    "
