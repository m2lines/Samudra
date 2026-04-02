#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --job-name=2026-04-02:samudra_llc:Agulhas_patch:temporal_stride=24
#SBATCH -N 1
#SBATCH --mem=200GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH -G h200:2
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@300
#SBATCH -o /orcd/home/002/codycruz/Ocean_Emulator/logs/%x-%j.out
#SBATCH -e /orcd/home/002/codycruz/Ocean_Emulator/logs/%x-%j.out

# load Python platform with PyTorch and CUDA support preinstalled
module load miniforge/24.3.0-0

# cd to correct directory
cd /orcd/home/002/codycruz/Ocean_Emulator

# activate uv environment for ocean_emulator
uv sync --dev

# reduce data fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# NCCL debugging
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_FR_BUFFER_SIZE="${TORCH_FR_BUFFER_SIZE:-1048576}"
export NCCL_DEBUG=INFO

# DDP params
DATA_NUM_WORKERS="${DATA_NUM_WORKERS:-4}"
PIN_MEM="${PIN_MEM:-false}"
DDP_BROADCAST_BUFFERS="${DDP_BROADCAST_BUFFERS:-false}"
DDP_TIMEOUT_MINUTES="${DDP_TIMEOUT_MINUTES:-60}"
DDP_MAX_DATA_WORKERS_PER_RANK="${DDP_MAX_DATA_WORKERS_PER_RANK:-12}"
CONCURRENT_COMPUTE="${CONCURRENT_COMPUTE:-false}"
LLC_FACE="${LLC_FACE:-1}"
LLC_I_START="${LLC_I_START:-2880}"
LLC_I_END="${LLC_I_END:-3600}"
LLC_J_START="${LLC_J_START:-720}"
LLC_J_END="${LLC_J_END:-1440}"
RESUME_CKPT_PATH="${RESUME_CKPT_PATH:-}"
FINETUNE="${FINETUNE:-false}"
RESET_OPTIMIZER_ON_RESUME="${RESET_OPTIMIZER_ON_RESUME:-true}"
RESET_SCHEDULER_ON_RESUME="${RESET_SCHEDULER_ON_RESUME:-true}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-}"

EPOCHS="${EPOCHS:-1}"
SAVE_FREQ="${SAVE_FREQ:-1}"
GPUS="${GPUS:-2}"

echo "======== train ocean_emulator samudra w/ ${GPUS} gpus on LLC4320 data ========"
echo "training for ${EPOCHS} total epochs and saving checkpoints every ${SAVE_FREQ}"
echo "using ${DATA_NUM_WORKERS} data workers and ${PIN_MEM} pin memory"
if [[ "${GPUS}" -gt 0 ]]; then
  echo "effective workers per rank (after trainer scaling): $((DATA_NUM_WORKERS / GPUS))"
fi
echo "using ddp_broadcast_buffers=${DDP_BROADCAST_BUFFERS} and ddp_timeout_minutes=${DDP_TIMEOUT_MINUTES}"
echo "using ddp_max_data_workers_per_rank=${DDP_MAX_DATA_WORKERS_PER_RANK}"
echo "using data.concurrent_compute=${CONCURRENT_COMPUTE}"
echo "using LLC face=${LLC_FACE}, i=[${LLC_I_START}:${LLC_I_END}), j=[${LLC_J_START}:${LLC_J_END})"

# Optional resume behavior:
# - RESUME_CKPT_PATH set + FINETUNE=false resumes optimizer/scheduler and starts at ckpt epoch + 1.
# - RESUME_CKPT_PATH set + FINETUNE=true loads model weights only and starts from epoch 1.
RESUME_ARGS=()
if [[ -n "${RESUME_CKPT_PATH}" ]]; then
  RESUME_ARGS+=(--resume_ckpt_path "${RESUME_CKPT_PATH}" --finetune "${FINETUNE}")
  RESUME_ARGS+=(--reset_optimizer_on_resume "${RESET_OPTIMIZER_ON_RESUME}")
  RESUME_ARGS+=(--reset_scheduler_on_resume "${RESET_SCHEDULER_ON_RESUME}")
  echo "resuming from checkpoint: ${RESUME_CKPT_PATH} (finetune=${FINETUNE})"
  echo "reset optimizer on resume: ${RESET_OPTIMIZER_ON_RESUME}"
  echo "reset scheduler on resume: ${RESET_SCHEDULER_ON_RESUME}"
fi

EXPERIMENT_ARGS=()
if [[ -n "${EXPERIMENT_NAME}" ]]; then
  EXPERIMENT_ARGS+=(--experiment.name "${EXPERIMENT_NAME}")
  echo "overriding experiment.name=${EXPERIMENT_NAME}"
fi
if [[ -n "${BASE_OUTPUT_DIR}" ]]; then
  EXPERIMENT_ARGS+=(--experiment.base_output_dir "${BASE_OUTPUT_DIR}")
  echo "overriding experiment.base_output_dir=${BASE_OUTPUT_DIR}"
fi

# Forward scheduler signals to torchrun so trainer can write emergency checkpoints.
TRAIN_PID=""
forward_signal() {
  local sig="$1"
  if [[ -n "${TRAIN_PID}" ]] && kill -0 "${TRAIN_PID}" 2>/dev/null; then
    echo "forwarding ${sig} to training process ${TRAIN_PID}"
    kill -s "${sig}" "${TRAIN_PID}"
  fi
}
trap 'forward_signal USR1' USR1
trap 'forward_signal TERM' TERM
trap 'forward_signal INT' INT

uv run python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node="${GPUS}" \
  -m ocean_emulators.train configs/samudra_llc/train_normal.yaml \
  --save_freq "${SAVE_FREQ}" \
  --epochs "${EPOCHS}" \
  --gradient_accumulation_steps 4 \
  --ddp_bucket_cap_mb 25 \
  --ddp_use_no_sync_for_accumulation true \
  --ddp_broadcast_buffers "${DDP_BROADCAST_BUFFERS}" \
  --ddp_timeout_minutes "${DDP_TIMEOUT_MINUTES}" \
  --ddp_max_data_workers_per_rank "${DDP_MAX_DATA_WORKERS_PER_RANK}" \
  --data.num_workers "${DATA_NUM_WORKERS}" \
  --data.concurrent_compute "${CONCURRENT_COMPUTE}" \
  --pin_mem "${PIN_MEM}" \
  --data.llc_face "${LLC_FACE}" \
  --data.llc_i_start "${LLC_I_START}" \
  --data.llc_i_end "${LLC_I_END}" \
  --data.llc_j_start "${LLC_J_START}" \
  --data.llc_j_end "${LLC_J_END}" \
  --experiment.data_root "/orcd/data/abodner/" \
  "${RESUME_ARGS[@]}" \
  "${EXPERIMENT_ARGS[@]}" &

TRAIN_PID=$!
wait "${TRAIN_PID}"
exit $?
