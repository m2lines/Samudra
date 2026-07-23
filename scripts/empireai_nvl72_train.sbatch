#!/bin/bash

# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

# Empire AI Beta NVL72 Slurm/Pyxis launcher for half-degree OM4 training.
#
# Smoke test, 2 GPUs on 1 host:
#   SMOKE=1 sbatch --account=YOUR_ACCOUNT --partition=beta --qos=test \
#     --nodes=1 --gres=gpu:2 --time=00:45:00 \
#     scripts/empireai_nvl72_train.sbatch
#
# Full run, 36 GPUs on 9 hosts:
#   sbatch --account=YOUR_ACCOUNT --partition=beta --qos=standard \
#     --nodes=9 --gres=gpu:4 --time=12:00:00 \
#     scripts/empireai_nvl72_train.sbatch
#
# Useful overrides:
#   export DATA_ROOT="${HOME}/data/om4_halfdeg"
#   export OUTPUT_BASE="${HOME}/runs"
#   export IMAGE_REF=ghcr.io/m2lines/ocean-emulator-physicsnemo@sha256:<digest>
#   export WANDB_API_KEY=...
#   export NCCL_SOCKET_IFNAME=bond0
#   export NCCL_IB_HCA="mlx5_0,mlx5_1,mlx5_4,mlx5_5"
#   export MELLANOX_VISIBLE_DEVICES=all
#   export EXTRA_ARGS="--learning_rate=0.0003"

#SBATCH --job-name=oe-halfdeg
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=beta
#SBATCH --account=nyu_beta_test
#SBATCH --qos=standard
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00

set -euo pipefail

SMOKE="${SMOKE:-0}"
CONFIG="${CONFIG:-configs/samudra_om4_v2/train.yaml}"
DATA_ROOT="${DATA_ROOT:-${HOME}/data/om4_halfdeg}"
OUTPUT_BASE="${OUTPUT_BASE:-${HOME}/runs}"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-${HOME}/.cache/samudra/${SLURM_JOB_ID:-manual}}"
MASTER_PORT="${MASTER_PORT:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
WANDB_ENTITY="${WANDB_ENTITY:-ocean_emulators}"
WANDB_PROJECT="${WANDB_PROJECT:-default}"

# Empire AI NVL72 nodes use ARM CPUs, and this Pyxis/Enroot stack does not
# import GHCR's OCI image indexes by tag. Pin the ARM64 manifest digest.
IMAGE_REF="${IMAGE_REF:-ghcr.io/m2lines/ocean-emulator-physicsnemo@sha256:d7cf627257df3e1fda74202435e00f55c15ebe2d2be9648f47daf31905f51bad}"

case "${SMOKE}" in
  0|1) ;;
  *)
    echo "SMOKE must be 0 or 1, got: ${SMOKE}" >&2
    exit 2
    ;;
esac

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "DATA_ROOT does not exist: ${DATA_ROOT}" >&2
  echo "Expected half-degree OM4 data at ${HOME}/data/om4_halfdeg by default." >&2
  exit 3
fi
for required in OM4.zarr OM4_means.zarr OM4_stds.zarr; do
  if [[ ! -e "${DATA_ROOT}/${required}" ]]; then
    echo "Missing ${DATA_ROOT}/${required}" >&2
    exit 3
  fi
done
mkdir -p "${OUTPUT_BASE}" "${DATA_CACHE_DIR}"

if [[ -z "${WANDB_MODE:-}" ]]; then
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    WANDB_MODE="online"
  else
    WANDB_MODE="disabled"
  fi
fi

gpus_on_node="${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-}}"
if [[ -z "${gpus_on_node}" ]]; then
  gpus_on_node="$(scontrol show job "${SLURM_JOB_ID}" | sed -n 's/.*gres:gpu:\\([0-9][0-9]*\\).*/\\1/p' | head -n 1)"
fi
GPUS_PER_NODE="$(printf '%s\n' "${gpus_on_node}" | grep -Eo '[0-9]+' | head -n 1 || true)"
if [[ -z "${GPUS_PER_NODE}" ]]; then
  echo "Could not determine GPUS_PER_NODE; set it explicitly." >&2
  exit 4
fi
export GPUS_PER_NODE

TOTAL_GPUS="$(( SLURM_NNODES * GPUS_PER_NODE ))"

if [[ -z "${OMP_NUM_THREADS:-}" ]]; then
  cpus_per_task="${SLURM_CPUS_PER_TASK:-16}"
  t="$(( cpus_per_task / GPUS_PER_NODE ))"
  if (( t < 1 )); then
    t=1
  fi
  export OMP_NUM_THREADS="${t}"
fi
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${OMP_NUM_THREADS}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${OMP_NUM_THREADS}}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export APPTAINERENV_PYTHONNOUSERSITE="${APPTAINERENV_PYTHONNOUSERSITE:-${PYTHONNOUSERSITE}}"
export SINGULARITYENV_PYTHONNOUSERSITE="${SINGULARITYENV_PYTHONNOUSERSITE:-${PYTHONNOUSERSITE}}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0,mlx5_1,mlx5_4,mlx5_5}"
export MELLANOX_VISIBLE_DEVICES="${MELLANOX_VISIBLE_DEVICES:-all}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

if [[ -z "${NAME:-}" ]]; then
  run_kind=""
  if [[ "${SMOKE}" == "1" ]]; then
    run_kind="-smoke"
  fi
  NAME="$(date +%Y-%m-%d-%H%M%S)-empireai${run_kind}"
fi
RUN_DIR="${OUTPUT_BASE}/${NAME}"
if [[ -e "${RUN_DIR}" ]]; then
  echo "Refusing to run: output dir already exists: ${RUN_DIR}" >&2
  exit 5
fi
mkdir -p "${RUN_DIR}"
export WANDB_DIR="${WANDB_DIR:-${RUN_DIR}/wandb}"
mkdir -p "${WANDB_DIR}"

CONFIG_IN_CONTAINER="${CONFIG}"
if [[ "${CONFIG_IN_CONTAINER}" != /* ]]; then
  CONFIG_IN_CONTAINER="/workspace/${CONFIG_IN_CONTAINER}"
fi

train_args=(
  "${CONFIG_IN_CONTAINER}"
  "--experiment.name=${NAME}"
  "--experiment.data_root=${DATA_ROOT}"
  "--experiment.base_output_dir=${OUTPUT_BASE}"
  "--experiment.wandb.mode=${WANDB_MODE}"
  "--experiment.wandb.entity=${WANDB_ENTITY}"
  "--experiment.wandb.project=${WANDB_PROJECT}"
  "--backend=nccl"
)

if [[ "${SMOKE}" == "1" ]]; then
  train_args+=(
    "--debug=true"
    "--epochs=1"
    "--batch_size=1"
    "--gradient_accumulation_steps=1"
    "--save_freq=1"
    "--steps=[1]"
    "--data_stride=[1]"
  )
fi

if [[ -n "${EXTRA_ARGS}" ]]; then
  read -r -a extra_args <<< "${EXTRA_ARGS}"
  train_args+=("${extra_args[@]}")
fi

MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
TORCHRUN_STANDALONE=0
if [[ -z "${MASTER_PORT}" && "${SLURM_NNODES}" == "1" ]]; then
  TORCHRUN_STANDALONE=1
elif [[ -z "${MASTER_PORT}" ]]; then
  MASTER_PORT="$((15000 + SLURM_JOB_ID % 40000))"
fi
export MASTER_ADDR MASTER_PORT TORCHRUN_STANDALONE

container_mounts="${DATA_ROOT}:${DATA_ROOT},${OUTPUT_BASE}:${OUTPUT_BASE},${DATA_CACHE_DIR}:/workspace/.data_cache"

echo "Smoke:         ${SMOKE}"
echo "Job id:        ${SLURM_JOB_ID}"
echo "Nodes:         ${SLURM_NNODES}"
echo "GPUs/node:     ${GPUS_PER_NODE}"
echo "Total GPUs:    ${TOTAL_GPUS}"
if [[ "${TORCHRUN_STANDALONE}" == "1" ]]; then
  echo "Rendezvous:    automatic free port (--standalone)"
else
  echo "Rendezvous:    ${MASTER_ADDR}:${MASTER_PORT}"
fi
echo "Image:         ${IMAGE_REF}"
echo "Config:        ${CONFIG_IN_CONTAINER}"
echo "Data root:     ${DATA_ROOT}"
echo "Output dir:    ${RUN_DIR}"
echo "W&B mode:      ${WANDB_MODE}"
echo "W&B entity:    ${WANDB_ENTITY}"
echo "W&B project:   ${WANDB_PROJECT}"
echo "NCCL socket IF:${NCCL_SOCKET_IFNAME}"
echo "NCCL IB HCAs:  ${NCCL_IB_HCA}"
echo "Mellanox devs: ${MELLANOX_VISIBLE_DEVICES}"

echo "Warming shared Pyxis image cache on ${MASTER_ADDR}..."
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --nodelist="${MASTER_ADDR}" \
  --container-image="${IMAGE_REF}" \
  --container-mount-home \
  --container-mounts="${container_mounts}" \
  true
echo "Pyxis image cache warmup complete."

srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
  --container-image="${IMAGE_REF}" \
  --container-mount-home \
  --container-mounts="${container_mounts}" \
  bash -lc '
    set -euo pipefail
    export PYTHONPATH=/workspace/src:${PYTHONPATH:-}
    if [[ ! -x /workspace/.venv/bin/python ]]; then
      echo "ERROR: /workspace/.venv/bin/python not found inside container" >&2
      exit 1
    fi
    if [[ ! -f "$1" ]]; then
      echo "ERROR: config not found inside container: $1" >&2
      exit 2
    fi
    torchrun_args=(
      --nnodes="${SLURM_NNODES}"
      --nproc_per_node="${GPUS_PER_NODE}"
      --node_rank="${SLURM_PROCID}"
    )
    if [[ "${TORCHRUN_STANDALONE}" == "1" ]]; then
      torchrun_args+=(--standalone)
    else
      torchrun_args+=(
        --master_addr="${MASTER_ADDR}"
        --master_port="${MASTER_PORT}"
      )
    fi
    exec /workspace/.venv/bin/python -m torch.distributed.run \
      "${torchrun_args[@]}" \
      -m samudra.train "$@"
  ' bash "${train_args[@]}"
