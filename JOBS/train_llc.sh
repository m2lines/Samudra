#!/bin/bash
#SBATCH -p pi_abodner
#SBATCH --job-name=2026-04-15-Samudra_LLC:revised_curriculum_1_test
#SBATCH -N 1
#SBATCH --mem=500GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --gres=gpu:4
#SBATCH --time=05-23:00:00
#SBATCH --signal=B:USR1@300
#SBATCH -o /orcd/home/002/codycruz/Ocean_Emulator/logs/%x-%j.out
#SBATCH -e /orcd/home/002/codycruz/Ocean_Emulator/logs/%x-%j.out

# load Python platform with PyTorch and CUDA support preinstalled
module load miniforge/24.3.0-0
module load cuda/13.1.0

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

# PROFILING
export NSYS_ARGS="--trace=cuda,nvtx,osrt --sample=cpu --delay=300 --duration=120"
NSYS_OUTPUT_DIR="/orcd/home/002/codycruz/Ocean_Emulator/logs/nsys"
mkdir -p "${NSYS_OUTPUT_DIR}"
PROFILER_CMD=()
if [[ -n "${NSYS_ARGS}" ]]; then
  if ! command -v nsys >/dev/null 2>&1; then
    echo "ERROR: NSYS_ARGS was set, but nsys is not available on PATH." >&2
    exit 1
  fi
  read -r -a nsys_args <<< "${NSYS_ARGS}"
  has_nsys_output=0
  for arg in "${nsys_args[@]}"; do
    case "${arg}" in
      -o|--output|-o?*|--output=*)
        has_nsys_output=1
        break
        ;;
    esac
  done
  PROFILER_CMD=(nsys profile "${nsys_args[@]}")
  if [[ "${has_nsys_output}" == "0" ]]; then
    PROFILER_CMD+=(
      -o "${NSYS_OUTPUT_DIR}/llc-${SLURM_JOB_ID:-manual}-node${SLURM_NODEID:-0}-proc${SLURM_PROCID:-0}"
    )
  fi
fi

# KNOBS

# GPUS WORKERS 
GPUS="${GPUS:-4}"
DATA_NUM_WORKERS="${DATA_NUM_WORKERS:-2}"

# DDP
PIN_MEM="${PIN_MEM:-false}"
DDP_BROADCAST_BUFFERS="${DDP_BROADCAST_BUFFERS:-false}"
DDP_TIMEOUT_MINUTES="${DDP_TIMEOUT_MINUTES:-300}"
DDP_MAX_DATA_WORKERS_PER_RANK="${DDP_MAX_DATA_WORKERS_PER_RANK:-12}"
CONCURRENT_COMPUTE="${CONCURRENT_COMPUTE:-false}"

# DATA
LLC_FACE="${LLC_FACE:-1}"
LLC_I_START="${LLC_I_START:-2880}"
LLC_I_END="${LLC_I_END:-3600}"
LLC_J_START="${LLC_J_START:-720}"
LLC_J_END="${LLC_J_END:-1440}"

# CHECKPOINTING FINETUNING
RESUME_CKPT_PATH="${RESUME_CKPT_PATH:-}" #.LOCAL/2026-04-03-CONT:[increase-step-test-suite]-WITH_temporal_stride=6,steps=4,2011-09-14-2012-01-01-RESUME/saved_nets/ckpt.pt}"
FINETUNE="${FINETUNE:-false}"
RESET_OPTIMIZER_ON_RESUME="${RESET_OPTIMIZER_ON_RESUME:-false}"
RESET_SCHEDULER_ON_RESUME="${RESET_SCHEDULER_ON_RESUME:-false}"

# NAME, DIRECTORY, EPOCHS, SAVE_FREQ
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}" # 2026-04-04-CONT:[increase-step-test-suite]-WITH_temporal_stride=6,steps=4,2012-01-01-2012-09-14-RESUME}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-}"
EPOCHS="${EPOCHS:-6}"
SAVE_FREQ="${SAVE_FREQ:-1}"

# OPTIMIZATION (LR + SCHEDULER)
LEARNING_RATE="${LEARNING_RATE:-0.0006}"
SCHEDULER_MODE="${SCHEDULER_MODE:-cosine}" # SCHEDULER_MODE: "cosine" (default) or "fixed" (no LR decay)
# If set while using cosine, stretches LR decay over a longer horizon than EPOCHS.
# Example: EPOCHS=6 and SCHEDULER_TARGET_EPOCHS=60 gives a much gentler decay.
SCHEDULER_TARGET_EPOCHS="${SCHEDULER_TARGET_EPOCHS:-60}"

# CURRICULUM
# list knobs should be passed like "[1]" or "[1, 2, 4]".
TEMPORAL_STRIDE="${TEMPORAL_STRIDE:-6}"
TEMPORAL_STRIDE_TRANSITION="${TEMPORAL_STRIDE_TRANSITION:-[]}"
STEPS="${STEPS:-[1,2,3,4,5,6]}"
STEP_TRANSITION="${STEP_TRANSITION:-[2,3,4,5,6]}"
DATA_STRIDE="${DATA_STRIDE:-[7]}"
HIST="${HIST:-1}"
GRADIENT_DETACH_INTERVAL="${GRADIENT_DETACH_INTERVAL:-4}"




echo "======== train ocean_emulator samudra w/ ${GPUS} gpus on LLC4320 data ========"
echo "training for ${EPOCHS} total epochs and saving checkpoints every ${SAVE_FREQ}"
echo "using ${DATA_NUM_WORKERS} data workers and ${PIN_MEM} pin memory"
if [[ "${GPUS}" -gt 0 ]]; then
  echo "effective workers per rank (after trainer scaling): $((DATA_NUM_WORKERS / GPUS))"
fi
echo "using ddp_broadcast_buffers=${DDP_BROADCAST_BUFFERS} and ddp_timeout_minutes=${DDP_TIMEOUT_MINUTES}"
echo "using ddp_max_data_workers_per_rank=${DDP_MAX_DATA_WORKERS_PER_RANK}"
echo "using data.concurrent_compute=${CONCURRENT_COMPUTE}"
echo "using optimization: learning_rate=${LEARNING_RATE}, scheduler_mode=${SCHEDULER_MODE}, scheduler_target_epochs=${SCHEDULER_TARGET_EPOCHS:-<default>}"
echo "using curriculum: data_stride=${DATA_STRIDE}, temporal_stride=${TEMPORAL_STRIDE}, steps=${STEPS}, step_transition=${STEP_TRANSITION}, temporal_stride_transition=${TEMPORAL_STRIDE_TRANSITION}, hist=${HIST}, grad-detach=${GRADIENT_DETACH_INTERVAL}"
echo "using data location: LLC face=${LLC_FACE}, i=[${LLC_I_START}:${LLC_I_END}), j=[${LLC_J_START}:${LLC_J_END})"

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

OPTIM_ARGS=(
  --learning_rate "${LEARNING_RATE}"
)

case "${SCHEDULER_MODE}" in
  fixed)
    OPTIM_ARGS+=(--scheduler null)
    ;;
  cosine)
    OPTIM_ARGS+=(--scheduler.type cosine)
    if [[ -n "${SCHEDULER_TARGET_EPOCHS}" ]]; then
      OPTIM_ARGS+=(--scheduler.target_epochs "${SCHEDULER_TARGET_EPOCHS}")
    fi
    ;;
  *)
    echo "ERROR: SCHEDULER_MODE must be 'cosine' or 'fixed' (got '${SCHEDULER_MODE}')." >&2
    exit 1
    ;;
esac

CURRICULUM_ARGS=(
  --data_stride "${DATA_STRIDE}"
  --temporal_stride "${TEMPORAL_STRIDE}"
  --steps "${STEPS}"
  --data.hist "${HIST}"
)

# pydantic-settings parses `--some_list "[]"` as `[""]` for list[int] fields.
# Omit transition flags entirely when they are empty; YAML defaults remain [].
STEP_TRANSITION_COMPACT="$(echo "${STEP_TRANSITION}" | tr -d '[:space:]')"
if [[ -n "${STEP_TRANSITION_COMPACT}" && "${STEP_TRANSITION_COMPACT}" != "[]" ]]; then
  CURRICULUM_ARGS+=(--step_transition "${STEP_TRANSITION}")
fi

TEMPORAL_STRIDE_TRANSITION_COMPACT="$(echo "${TEMPORAL_STRIDE_TRANSITION}" | tr -d '[:space:]')"
if [[ -n "${TEMPORAL_STRIDE_TRANSITION_COMPACT}" && "${TEMPORAL_STRIDE_TRANSITION_COMPACT}" != "[]" ]]; then
  CURRICULUM_ARGS+=(--temporal_stride_transition "${TEMPORAL_STRIDE_TRANSITION}")
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
  -m ocean_emulators.train configs/samudra_llc/train.yaml \
  --save_freq "${SAVE_FREQ}" \
  --epochs "${EPOCHS}" \
  "${OPTIM_ARGS[@]}" \
  "${CURRICULUM_ARGS[@]}" \
  --model.gradient_detach_interval "${GRADIENT_DETACH_INTERVAL}" \
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
