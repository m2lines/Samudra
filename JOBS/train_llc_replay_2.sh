#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --job-name=2026-07-18:samudra_llc:rb-Agulhas-strides=1-pred_resid-eager-4-LOSS-RESTART
#SBATCH -x node4100,node3401,node3000
#SBATCH -N 1
#SBATCH --mem=254GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH -G h200:1
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@300
#SBATCH -o /orcd/home/002/codycruz/Ocean_Emulator/logs/%x-%j.out
#SBATCH -e /orcd/home/002/codycruz/Ocean_Emulator/logs/%x-%j.out

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-<unset>}"
echo "SLURMD_NODENAME=${SLURMD_NODENAME:-<unset>}"
echo "hostname=$(hostname)"

module load miniforge/24.3.0-0
module load cuda/13.1.0

cd /orcd/home/002/codycruz/Ocean_Emulator

PROJECT_SITE_PACKAGES="/orcd/home/002/codycruz/Ocean_Emulator/.venv/lib/python3.11/site-packages"
export PYTHONPATH="/orcd/home/002/codycruz/Ocean_Emulator/src:${PROJECT_SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="${PYTHON_BIN:-/orcd/home/002/codycruz/Ocean_Emulator/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: expected Python 3.11 environment at ${PYTHON_BIN}, but it is not executable." >&2
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_FR_BUFFER_SIZE="${TORCH_FR_BUFFER_SIZE:-1048576}"
export NCCL_DEBUG=INFO

# PROFILING
NSYS_ENABLE="${NSYS_ENABLE:-false}"
export NSYS_ARGS="${NSYS_ARGS:---trace=cuda,nvtx,osrt --sample=cpu --delay=360 --duration=600 --force-overwrite=true}"
NSYS_OUTPUT_DIR="/orcd/home/002/codycruz/Ocean_Emulator/logs/nsys"
mkdir -p "${NSYS_OUTPUT_DIR}"
PROFILER_CMD=()
if [[ "${NSYS_ENABLE}" == "true" ]]; then
  if ! command -v nsys >/dev/null 2>&1; then
    echo "ERROR: NSYS_ENABLE=true, but nsys is not available on PATH." >&2
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
  echo "NSYS profiling enabled: ${PROFILER_CMD[*]}"
else
  echo "NSYS profiling disabled (set NSYS_ENABLE=true to enable; NSYS_ARGS='${NSYS_ARGS}')"
fi

# GPUS / DATA WORKERS
GPUS="${GPUS:-1}"
DATA_NUM_WORKERS="${DATA_NUM_WORKERS:-6}"
DATA_PREFETCH_FACTOR="${DATA_PREFETCH_FACTOR:-6}"
BLOSC_THREADS="${BLOSC_THREADS:-1}"
export OCEAN_BLOSC_THREADS="${OCEAN_BLOSC_THREADS:-${BLOSC_THREADS}}"
SURFACE_SNAPSHOT="${SURFACE_SNAPSHOT:-true}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
PIN_MEM="${PIN_MEM:-true}"
CONCURRENT_COMPUTE="${CONCURRENT_COMPUTE:-false}"

# MODEL
PAD="${PAD:-constant}"
NUM_HALO="${NUM_HALO:-4}"
NUM_SPONGE="${NUM_SPONGE:-12}"
PRED_RESIDUALS="${PRED_RESIDUALS:-true}"

# DDP
DDP_BROADCAST_BUFFERS="${DDP_BROADCAST_BUFFERS:-false}"
DDP_TIMEOUT_MINUTES="${DDP_TIMEOUT_MINUTES:-300}"
DDP_MAX_DATA_WORKERS_PER_RANK="${DDP_MAX_DATA_WORKERS_PER_RANK:-12}"

# DATA
LLC_FACE="${LLC_FACE:-1}"
LLC_I_START="${LLC_I_START:-2880}"
LLC_I_END="${LLC_I_END:-3600}"
LLC_J_START="${LLC_J_START:-720}"
LLC_J_END="${LLC_J_END:-1440}"
DATA_LOCATION_OVERRIDE="${DATA_LOCATION_OVERRIDE:-}"
DATA_STRIDE="${DATA_STRIDE:-[1]}"
TEMPORAL_STRIDE="${TEMPORAL_STRIDE:-1}"
TEMPORAL_STRIDE_TRANSITION="${TEMPORAL_STRIDE_TRANSITION:-[]}"
HIST="${HIST:-0}"

# CHECKPOINTING / RESUME
RESUME_CKPT_PATH="${RESUME_CKPT_PATH:-/orcd/data/abodner/002/cody/overflow/wandb_overflow/rb/2026-07-16:samudra_llc:rb-Agulhas-strides=1-pred_resid-eager-3-LOSS-RESTART-18075148/saved_nets/ckpt_emergency.pt}"
FINETUNE="${FINETUNE:-false}"
RESET_OPTIMIZER_ON_RESUME="${RESET_OPTIMIZER_ON_RESUME:-false}"
RESET_SCHEDULER_ON_RESUME="${RESET_SCHEDULER_ON_RESUME:-false}"

# NAME / EPOCHS / SAVE_FREQ
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${SLURM_JOB_NAME:-$(basename "$0" .sh)}}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-}"
EPOCHS="${EPOCHS:-50}"
SAVE_FREQ="${SAVE_FREQ:-5}"
EMERGENCY_CHECKPOINT_INTERVAL_MINUTES="${EMERGENCY_CHECKPOINT_INTERVAL_MINUTES:-30}"
EXPERIMENT_NAME="${EXPERIMENT_NAME}${SLURM_JOB_ID:+-${SLURM_JOB_ID}}"

# OPTIMIZATION
LEARNING_RATE="${LEARNING_RATE:-0.0006}"
SCHEDULER_MODE="${SCHEDULER_MODE:-cosine}"
SCHEDULER_TARGET_EPOCHS="${SCHEDULER_TARGET_EPOCHS:-70}"
LR_MULTIPLIERS="${LR_MULTIPLIERS:-[1.0]}"
LR_MULTIPLIER_TRANSITION="${LR_MULTIPLIER_TRANSITION:-[]}"

# REPLAY BUFFER
REPLAY_ENABLED="${REPLAY_ENABLED:-true}"
REPLAY_BUFFER_SIZE="${REPLAY_BUFFER_SIZE:-32}"
REPLAY_REFRESH_EVERY_N_MICROBATCHES="${REPLAY_REFRESH_EVERY_N_MICROBATCHES:-[8,12,16,20,24,28,32,36,40,44]}"
REPLAY_REFRESH_EVERY_N_MICROBATCHES_TRANSITION="${REPLAY_REFRESH_EVERY_N_MICROBATCHES_TRANSITION:-[6, 11, 16, 21, 26, 31, 36, 41, 46]}"
REPLAY_STEPS_PER_EPOCH="${REPLAY_STEPS_PER_EPOCH:-8760}"
REPLAY_MAX_LEAD_STEPS="${REPLAY_MAX_LEAD_STEPS:-[4, 8, 12, 16, 20, 24, 28, 32, 34, 40]}"
REPLAY_MAX_LEAD_TRANSITION="${REPLAY_MAX_LEAD_TRANSITION:-[6, 11, 16, 21, 26, 31, 36, 41, 46]}"
REPLAY_CHECKPOINT_BUFFER="${REPLAY_CHECKPOINT_BUFFER:-true}"

echo "======== train ocean_emulator replay samudra w/ ${GPUS} gpus on LLC4320 data ========"
echo "training for ${EPOCHS} total epochs and saving checkpoints every ${SAVE_FREQ}"
echo "saving overwrite emergency checkpoints every ${EMERGENCY_CHECKPOINT_INTERVAL_MINUTES} minutes"
echo "using ${DATA_NUM_WORKERS} data workers, prefetch_factor=${DATA_PREFETCH_FACTOR}, blosc_threads=${OCEAN_BLOSC_THREADS}, pin_mem=${PIN_MEM}"
if [[ "${GPUS}" -gt 0 ]]; then
  echo "effective workers per rank (after trainer scaling): $((DATA_NUM_WORKERS / GPUS))"
fi
echo "using validation surface_snapshot=${SURFACE_SNAPSHOT}"
echo "using ddp_broadcast_buffers=${DDP_BROADCAST_BUFFERS}, ddp_timeout_minutes=${DDP_TIMEOUT_MINUTES}, ddp_max_data_workers_per_rank=${DDP_MAX_DATA_WORKERS_PER_RANK}"
echo "using optimization: learning_rate=${LEARNING_RATE}, scheduler_mode=${SCHEDULER_MODE}, scheduler_target_epochs=${SCHEDULER_TARGET_EPOCHS:-<default>}"
echo "using lr multipliers: lr_multipliers=${LR_MULTIPLIERS}, lr_multiplier_transition=${LR_MULTIPLIER_TRANSITION}"
echo "using replay data: data_stride=${DATA_STRIDE}, temporal_stride=${TEMPORAL_STRIDE}, temporal_stride_transition=${TEMPORAL_STRIDE_TRANSITION}, hist=${HIST}"
echo "using replay: enabled=${REPLAY_ENABLED}, buffer_size=${REPLAY_BUFFER_SIZE}, refresh_every_n_microbatches=${REPLAY_REFRESH_EVERY_N_MICROBATCHES}, refresh_every_n_microbatches_transition=${REPLAY_REFRESH_EVERY_N_MICROBATCHES_TRANSITION}, steps_per_epoch=${REPLAY_STEPS_PER_EPOCH}, max_lead_steps=${REPLAY_MAX_LEAD_STEPS}, max_lead_transition=${REPLAY_MAX_LEAD_TRANSITION}, checkpoint_buffer=${REPLAY_CHECKPOINT_BUFFER}"
echo "using data location: LLC face=${LLC_FACE}, i=[${LLC_I_START}:${LLC_I_END}), j=[${LLC_J_START}:${LLC_J_END})"
echo "using padding: pad=${PAD}, num_halo=${NUM_HALO}, num_sponge=${NUM_SPONGE}"
echo "predicting field or residual: pred_residual=${PRED_RESIDUALS}"
echo "using batch_size=${BATCH_SIZE}, gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}, effective_batch_size=$((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
if [[ -n "${DATA_LOCATION_OVERRIDE}" ]]; then
  echo "overriding data.data_location=${DATA_LOCATION_OVERRIDE}"
fi

RESUME_ARGS=()
if [[ -n "${RESUME_CKPT_PATH}" ]]; then
  RESUME_ARGS+=(--resume_ckpt_path "${RESUME_CKPT_PATH}" --finetune "${FINETUNE}")
  RESUME_ARGS+=(--reset_optimizer_on_resume "${RESET_OPTIMIZER_ON_RESUME}")
  RESUME_ARGS+=(--reset_scheduler_on_resume "${RESET_SCHEDULER_ON_RESUME}")
  echo "resuming from checkpoint: ${RESUME_CKPT_PATH} (finetune=${FINETUNE})"
  echo "reset optimizer on resume: ${RESET_OPTIMIZER_ON_RESUME}"
  echo "reset scheduler on resume: ${RESET_SCHEDULER_ON_RESUME}"
fi

EXPERIMENT_ARGS=(--experiment.name "${EXPERIMENT_NAME}")
echo "overriding experiment.name=${EXPERIMENT_NAME}"
if [[ -n "${BASE_OUTPUT_DIR}" ]]; then
  EXPERIMENT_ARGS+=(--experiment.base_output_dir "${BASE_OUTPUT_DIR}")
  echo "overriding experiment.base_output_dir=${BASE_OUTPUT_DIR}"
fi

OPTIM_ARGS=(--learning_rate "${LEARNING_RATE}")
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

REPLAY_DATA_ARGS=(
  --data_stride "${DATA_STRIDE}"
  --temporal_stride "${TEMPORAL_STRIDE}"
  --data.hist "${HIST}"
  --lr_multipliers "${LR_MULTIPLIERS}"
)

TEMPORAL_STRIDE_TRANSITION_COMPACT="$(echo "${TEMPORAL_STRIDE_TRANSITION}" | tr -d '[:space:]')"
if [[ -n "${TEMPORAL_STRIDE_TRANSITION_COMPACT}" && "${TEMPORAL_STRIDE_TRANSITION_COMPACT}" != "[]" ]]; then
  REPLAY_DATA_ARGS+=(--temporal_stride_transition "${TEMPORAL_STRIDE_TRANSITION}")
fi

LR_MULTIPLIER_TRANSITION_COMPACT="$(echo "${LR_MULTIPLIER_TRANSITION}" | tr -d '[:space:]')"
if [[ -n "${LR_MULTIPLIER_TRANSITION_COMPACT}" && "${LR_MULTIPLIER_TRANSITION_COMPACT}" != "[]" ]]; then
  REPLAY_DATA_ARGS+=(--lr_multiplier_transition "${LR_MULTIPLIER_TRANSITION}")
fi

REPLAY_ARGS=(
  --replay.enabled "${REPLAY_ENABLED}"
  --replay.buffer_size "${REPLAY_BUFFER_SIZE}"
  --replay.refresh_every_n_microbatches "${REPLAY_REFRESH_EVERY_N_MICROBATCHES}"
  --replay.steps_per_epoch "${REPLAY_STEPS_PER_EPOCH}"
  --replay.max_lead_steps "${REPLAY_MAX_LEAD_STEPS}"
  --replay.max_lead_transition "${REPLAY_MAX_LEAD_TRANSITION}"
  --replay.checkpoint_buffer "${REPLAY_CHECKPOINT_BUFFER}"
)

REPLAY_REFRESH_TRANSITION_COMPACT="$(echo "${REPLAY_REFRESH_EVERY_N_MICROBATCHES_TRANSITION}" | tr -d '[:space:]')"
if [[ -n "${REPLAY_REFRESH_TRANSITION_COMPACT}" && "${REPLAY_REFRESH_TRANSITION_COMPACT}" != "[]" ]]; then
  REPLAY_ARGS+=(--replay.refresh_every_n_microbatches_transition "${REPLAY_REFRESH_EVERY_N_MICROBATCHES_TRANSITION}")
fi

DATA_OVERRIDE_ARGS=()
if [[ -n "${DATA_LOCATION_OVERRIDE}" ]]; then
  DATA_OVERRIDE_ARGS+=(--data.data_location "${DATA_LOCATION_OVERRIDE}")
fi

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

"${PROFILER_CMD[@]}" "${PYTHON_BIN}" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node="${GPUS}" \
  -m ocean_emulators.train configs/samudra_llc/train_replay.yaml \
  --save_freq "${SAVE_FREQ}" \
  --epochs "${EPOCHS}" \
  --emergency_checkpoint_interval_minutes "${EMERGENCY_CHECKPOINT_INTERVAL_MINUTES}" \
  "${OPTIM_ARGS[@]}" \
  "${REPLAY_DATA_ARGS[@]}" \
  "${REPLAY_ARGS[@]}" \
  --model.pad "${PAD}" \
  --model.num_halo "${NUM_HALO}" \
  --model.num_sponge "${NUM_SPONGE}" \
  --model.pred_residuals "${PRED_RESIDUALS}" \
  --batch_size "${BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --ddp_bucket_cap_mb 25 \
  --ddp_use_no_sync_for_accumulation true \
  --ddp_broadcast_buffers "${DDP_BROADCAST_BUFFERS}" \
  --ddp_timeout_minutes "${DDP_TIMEOUT_MINUTES}" \
  --ddp_max_data_workers_per_rank "${DDP_MAX_DATA_WORKERS_PER_RANK}" \
  --surface_snapshot "${SURFACE_SNAPSHOT}" \
  --data.num_workers "${DATA_NUM_WORKERS}" \
  --data.prefetch_factor "${DATA_PREFETCH_FACTOR}" \
  --data.concurrent_compute "${CONCURRENT_COMPUTE}" \
  --pin_mem "${PIN_MEM}" \
  --data.llc_face "${LLC_FACE}" \
  --data.llc_i_start "${LLC_I_START}" \
  --data.llc_i_end "${LLC_I_END}" \
  --data.llc_j_start "${LLC_J_START}" \
  --data.llc_j_end "${LLC_J_END}" \
  --experiment.data_root "/orcd/data/abodner/" \
  "${DATA_OVERRIDE_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  "${EXPERIMENT_ARGS[@]}" &

TRAIN_PID=$!
wait "${TRAIN_PID}"
exit $?
