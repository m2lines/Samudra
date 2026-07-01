#!/bin/bash
#SBATCH -p pi_abodner
#SBATCH --job-name=samudra_llc_diagnose
#SBATCH -N 1
#SBATCH --mem=600GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --gres=gpu:3
#SBATCH --time=00-08:00:00
#SBATCH -o /orcd/home/002/codycruz/Ocean_Emulator/logs/%x-%j.out
#SBATCH -e /orcd/home/002/codycruz/Ocean_Emulator/logs/%x-%j.out

set -euo pipefail

# ---------- Adjustable knobs (override via env vars at submit time) ----------
PROJECT_ROOT="${PROJECT_ROOT:-/orcd/home/002/codycruz/Ocean_Emulator}"
CONFIG_PATH="${CONFIG_PATH:-configs/samudra_llc/train.yaml}"
DATA_ROOT="${DATA_ROOT:-/orcd/data/abodner/}"

GPUS="${GPUS:-3}"
EPOCHS="${EPOCHS:-1}"
SAVE_FREQ="${SAVE_FREQ:-1}"

GRAD_ACC="${GRAD_ACC:-4}"
DDP_BUCKET_CAP_MB="${DDP_BUCKET_CAP_MB:-25}"
DDP_USE_NO_SYNC="${DDP_USE_NO_SYNC:-true}"
DDP_BROADCAST_BUFFERS="${DDP_BROADCAST_BUFFERS:-false}"
DDP_STATIC_GRAPH="${DDP_STATIC_GRAPH:-false}"
DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-false}"

DATA_NUM_WORKERS="${DATA_NUM_WORKERS:-18}"
PIN_MEM="${PIN_MEM:-true}"
CONCURRENT_COMPUTE="${CONCURRENT_COMPUTE:-false}"

WANDB_MODE="${WANDB_MODE:-disabled}"
NCCL_DEBUG_LEVEL="${NCCL_DEBUG_LEVEL:-INFO}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-$(date +%Y-%m-%d)-samudra_llc:diagnose:thermo_fields-Qnet-forcings,all_depths,extent=719}"
EXPERIMENT_NAME="${EXPERIMENT_NAME}${SLURM_JOB_ID:+-${SLURM_JOB_ID}}"

# load Python platform with PyTorch and CUDA support preinstalled
module load miniforge/24.3.0-0

cd "${PROJECT_ROOT}"
uv sync --dev

# runtime env
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

# NCCL debugging
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
export TORCH_FR_BUFFER_SIZE="${TORCH_FR_BUFFER_SIZE:-1048576}"
export NCCL_DEBUG="${NCCL_DEBUG_LEVEL}"

echo "======== diagnose ocean_emulator training ========"
echo "GPUS=${GPUS} EPOCHS=${EPOCHS} SAVE_FREQ=${SAVE_FREQ}"
echo "GRAD_ACC=${GRAD_ACC} DDP_BUCKET_CAP_MB=${DDP_BUCKET_CAP_MB}"
echo "DDP_USE_NO_SYNC=${DDP_USE_NO_SYNC} DDP_BROADCAST_BUFFERS=${DDP_BROADCAST_BUFFERS}"
echo "DDP_STATIC_GRAPH=${DDP_STATIC_GRAPH} DDP_FIND_UNUSED_PARAMETERS=${DDP_FIND_UNUSED_PARAMETERS}"
echo "DATA_NUM_WORKERS=${DATA_NUM_WORKERS} PIN_MEM=${PIN_MEM} CONCURRENT_COMPUTE=${CONCURRENT_COMPUTE}"
echo "WANDB_MODE=${WANDB_MODE}"
echo "EXPERIMENT_NAME=${EXPERIMENT_NAME}"

CMD=(
  uv run python -m torch.distributed.run
  --standalone --nnodes=1 --nproc_per_node="${GPUS}"
  -m ocean_emulators.train "${CONFIG_PATH}"
  --save_freq "${SAVE_FREQ}"
  --epochs "${EPOCHS}"
  --gradient_accumulation_steps "${GRAD_ACC}"
  --ddp_bucket_cap_mb "${DDP_BUCKET_CAP_MB}"
  --ddp_use_no_sync_for_accumulation "${DDP_USE_NO_SYNC}"
  --ddp_broadcast_buffers "${DDP_BROADCAST_BUFFERS}"
  --ddp_static_graph "${DDP_STATIC_GRAPH}"
  --ddp_find_unused_parameters "${DDP_FIND_UNUSED_PARAMETERS}"
  --data.num_workers "${DATA_NUM_WORKERS}"
  --pin_mem "${PIN_MEM}"
  --data.concurrent_compute "${CONCURRENT_COMPUTE}"
  --experiment.wandb.mode "${WANDB_MODE}"
  --experiment.name "${EXPERIMENT_NAME}"
  --experiment.data_root "${DATA_ROOT}"
)

if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=(${EXTRA_ARGS})
  CMD+=("${EXTRA_ARR[@]}")
fi

"${CMD[@]}"
