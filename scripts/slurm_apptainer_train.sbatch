#!/bin/bash
# Slurm + Apptainer harness for distributed training, modeled after `skypilot/train.sky.yaml`.
#
# Submit with env vars (example):
#   export CONFIG=configs/.../train.yaml
#   export NAME=my_run
#   export ARGS="--epochs=2 --batch_size=1"
#   export WANDB_API_KEY=...   # optional; if unset, wandb is disabled
#   sbatch --nodes=1 --gres=gpu:rtx6000:8 scripts/slurm_apptainer_train.sbatch
#
# Multi-node example (requires the partition to actually have >=N nodes available):
#   sbatch --nodes=2 --gres=gpu:rtx6000:8 scripts/slurm_apptainer_train.sbatch
#
# Container selection (pick one):
#   export IMAGE_REF=ghcr.io/open-athena/ocean-emulator-physicsnemo:25.11-<hash>
#   export CONTAINER_HASH=<git sha>   # expands to 25.11-<sha>
#   export CONTAINER_TAG=25.11-latest
#
# Private GHCR pulls (if needed):
#   export GHCR_USERNAME=...
#   export GHCR_TOKEN=...

#SBATCH --job-name=oe-train
#SBATCH --output=slurm-%j.out
#SBATCH --account=torch_pr_347_courant
#SBATCH --partition=rtx6000_lzanna
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
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
NAME="${NAME:-${RUN_NAME:-}}"
ARGS="${ARGS:-}"
if [[ -z "${CONFIG}" || -z "${NAME}" ]]; then
  echo "Missing required env vars. Need CONFIG and NAME (or RUN_NAME)." >&2
  exit 2
fi

# Config path inside the container image. We keep the env var as-is for
# backwards compatibility, but resolve relative paths under /workspace/
# (the image's working directory where configs/ is copied).
CONFIG_IN_CONTAINER="${CONFIG}"
if [[ "${CONFIG_IN_CONTAINER}" != /* ]]; then
  CONFIG_IN_CONTAINER="/workspace/${CONFIG_IN_CONTAINER}"
fi

# Data + outputs (host paths; will be bind-mounted into the container at same paths).
DATA_ROOT="${DATA_ROOT:-/scratch/jr7309/data/om4_onedeg_v3}"
OUTPUT_BASE="${OUTPUT_BASE:-/scratch/jr7309/runs}"

mkdir -p "${OUTPUT_BASE}"
RUN_DIR="${OUTPUT_BASE}/${NAME}"
if [[ -e "${RUN_DIR}" ]]; then
  echo "Refusing to run: output dir already exists: ${RUN_DIR}" >&2
  echo "Pick a unique NAME (we recommend a date prefix)." >&2
  exit 4
fi
mkdir -p "${RUN_DIR}"

# W&B mode defaults: online if key provided, else disabled.
WANDB_MODE="${WANDB_MODE:-}"
if [[ -z "${WANDB_MODE}" ]]; then
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    WANDB_MODE="online"
  else
    WANDB_MODE="disabled"
  fi
fi
if [[ "${WANDB_MODE}" == "offline" ]]; then
  # TrainConfig.experiment.wandb.mode only accepts "online" or "disabled".
  WANDB_MODE="disabled"
fi

# If WANDB_DIR is set, ensure it's writable on the host before entering the container.
if [[ -n "${WANDB_DIR:-}" ]]; then
  mkdir -p "${WANDB_DIR}"
fi

# Container selection (same conventions as scripts/slurm_apptainer_gpu_tests.sbatch).
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

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_DIR}"

SIF_DIR="${SIF_DIR:-${REPO_DIR}/.apptainer-images}"
mkdir -p "${SIF_DIR}"
TAG_SAFE="$(echo "${CONTAINER_TAG}" | tr '/:@' '---')"
SIF_PATH="${SIF_PATH:-${SIF_DIR}/${CONTAINER_REPO}-${TAG_SAFE}.sif}"

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
echo "W&B mode:   ${WANDB_MODE}"
echo "Image ref:  ${IMAGE_REF}"
echo "SIF path:   ${SIF_PATH}"

if [[ ! -s "${SIF_PATH}" ]]; then
  echo "SIF not found locally. Pulling from registry..."
  "${APPTAINER_BIN}" pull "${SIF_PATH}" "docker://${IMAGE_REF}"
else
  echo "Using existing SIF."
fi

# Distributed rendezvous info (shared by all nodes).
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-8008}"
export MASTER_ADDR MASTER_PORT

# This harness launches one torchrun per node via srun. Each torchrun spawns one
# process per GPU on that node, and sets RANK/LOCAL_RANK/WORLD_SIZE via env://.
#
# Resource requests must match reality:
# - If you request 8 GPUs per node, torchrun will use 8 processes per node.
GPUS_PER_NODE="${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-}}"
if [[ -z "${GPUS_PER_NODE}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS_PER_NODE="$(nvidia-smi -L | wc -l | tr -d ' ')"
  else
    echo "Could not determine GPUS_PER_NODE (SLURM_GPUS_ON_NODE unset and nvidia-smi missing)." >&2
    exit 3
  fi
fi

# Avoid oversubscribing CPU threads when torchrun spawns one process per GPU.
if [[ -z "${OMP_NUM_THREADS:-}" ]]; then
  if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
    t="$(( SLURM_CPUS_PER_TASK / GPUS_PER_NODE ))"
    if (( t < 1 )); then
      t=1
    fi
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
        echo \"ERROR: config not found inside container: ${CONFIG_IN_CONTAINER}\" >&2
        echo \"Hint: rebuild/pull a container tag that includes /workspace/configs.\" >&2
        exit 2
      fi
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
