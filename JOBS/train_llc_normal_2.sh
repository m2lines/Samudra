#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --account=mit_amf_standard_gpu
#SBATCH --qos=mit_amf_standard_gpu
#SBATCH --job-name=2026-06-08:samudra_llc:B-9
#SBATCH --exclude=node3401,node3400,node4100
#SBATCH -N 1
#SBATCH --mem=254GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH -G h200:1
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@300
#SBATCH -o /orcd/home/002/codycruz/Ocean_Emulator/logs/%x-%j.out
#SBATCH -e /orcd/home/002/codycruz/Ocean_Emulator/logs/%x-%j.out

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-<unset>}"
echo "SLURMD_NODENAME=${SLURMD_NODENAME:-<unset>}"
echo "hostname=$(hostname)"

# DDP# load Python platform with PyTorch and CUDA support preinstalled
module load miniforge/24.3.0-0
module load cuda/13.1.0

# cd to correct directory
cd /orcd/home/002/codycruz/Ocean_Emulator

# Use the already-built project environment directly instead of `uv run`, which
# still mutates the shared `.venv` on this filesystem even with `--no-sync`.
PROJECT_SITE_PACKAGES="/orcd/home/002/codycruz/Ocean_Emulator/.venv/lib/python3.11/site-packages"
export PYTHONPATH="/orcd/home/002/codycruz/Ocean_Emulator/src:${PROJECT_SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="${PYTHON_BIN:-/orcd/home/002/codycruz/Ocean_Emulator/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: expected Python 3.11 environment at ${PYTHON_BIN}, but it is not executable." >&2
  echo "If this node lacks /usr/bin/python3.11, recreate .venv with a portable Python 3.11 install or a venv built with --copies." >&2
  exit 1
fi

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
GPUS="${GPUS:-1}"
DATA_NUM_WORKERS="${DATA_NUM_WORKERS:-6}"
DATA_PREFETCH_FACTOR="${DATA_PREFETCH_FACTOR:-6}"
TRAIN_SHUFFLE="${TRAIN_SHUFFLE:-true}"
SURFACE_SNAPSHOT="${SURFACE_SNAPSHOT:-true}"
PAD="${PAD:-constant}"
NUM_HALO="${NUM_HALO:-4}"
NUM_SPONGE="${NUM_SPONGE:-12}"

# DDP
PIN_MEM="${PIN_MEM:-true}"
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
DATA_LOCATION_OVERRIDE="${DATA_LOCATION_OVERRIDE:-}"

# CHECKPOINTING FINETUNING
RESUME_CKPT_PATH="${RESUME_CKPT_PATH:-/home/codycruz/Ocean_Emulator/.LOCAL/2026-06-07:samudra_llc:B-8-15594577/saved_nets/ckpt_emergency.pt}" #/home/codycruz/Ocean_Emulator/.LOCAL/2026-04-24-Samudra_LLC:config_tests_experiment_6_multi_epochs/saved_nets/ckpt_6.pt
FINETUNE="${FINETUNE:-false}"
RESET_OPTIMIZER_ON_RESUME="${RESET_OPTIMIZER_ON_RESUME:-false}"
RESET_SCHEDULER_ON_RESUME="${RESET_SCHEDULER_ON_RESUME:-false}"

# NAME, DIRECTORY, EPOCHS, SAVE_FREQ
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${SLURM_JOB_NAME:-$(basename "$0" .sh)}}" # 2026-04-04-CONT:[increase-step-test-suite]-WITH_temporal_stride=6,steps=4,2012-01-01-2012-09-14-RESUME}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-}"
EPOCHS="${EPOCHS:-72}"
SAVE_FREQ="${SAVE_FREQ:-1}"
EMERGENCY_CHECKPOINT_INTERVAL_MINUTES="${EMERGENCY_CHECKPOINT_INTERVAL_MINUTES:-30}"
EXPERIMENT_NAME="${EXPERIMENT_NAME}${SLURM_JOB_ID:+-${SLURM_JOB_ID}}"

# OPTIMIZATION (LR + SCHEDULER)
LEARNING_RATE="${LEARNING_RATE:-0.0006}"
SCHEDULER_MODE="${SCHEDULER_MODE:-cosine}" # SCHEDULER_MODE: "cosine" (default) or "fixed" (no LR decay)
# If set while using cosine, stretches LR decay over a longer horizon than EPOCHS.
# Example: EPOCHS=6 and SCHEDULER_TARGET_EPOCHS=60 gives a much gentler decay.
SCHEDULER_TARGET_EPOCHS="${SCHEDULER_TARGET_EPOCHS:-90}"
LR_MULTIPLIERS="${LR_MULTIPLIERS:-[1.0, 0.67, 0.85, 1.0, 0.67, 0.85, 1.0, 0.67, 0.85, 1.0, 0.67, 0.85, 1.0, 0.67, 0.85, 1.0]}"
LR_MULTIPLIER_TRANSITION="${LR_MULTIPLIER_TRANSITION:-[13, 16, 19, 25, 28, 31, 37, 40, 43, 49, 52, 55, 61, 64, 67]}"

# CURRICULUM
# list knobs should be passed like "[1]" or "[1, 2, 4]".
TEMPORAL_STRIDE="${TEMPORAL_STRIDE:-3}"
TEMPORAL_STRIDE_TRANSITION="${TEMPORAL_STRIDE_TRANSITION:-[]}"
STEPS="${STEPS:-[2, 3, 4, 5, 6, 7]}"
STEP_TRANSITION="${STEP_TRANSITION:-[13,25,37,49,61]}" 
DATA_STRIDE="${DATA_STRIDE:-[3]}"
HIST="${HIST:-0}"
GRADIENT_DETACH_INTERVAL="${GRADIENT_DETACH_INTERVAL:-3}"



echo "======== train ocean_emulator samudra w/ ${GPUS} gpus on LLC4320 data ========"
echo "training for ${EPOCHS} total epochs and saving checkpoints every ${SAVE_FREQ}"
echo "saving overwrite emergency checkpoints every ${EMERGENCY_CHECKPOINT_INTERVAL_MINUTES} minutes"
echo "using ${DATA_NUM_WORKERS} data workers and ${PIN_MEM} pin memory"
if [[ "${GPUS}" -gt 0 ]]; then
  echo "effective workers per rank (after trainer scaling): $((DATA_NUM_WORKERS / GPUS))"
fi
echo "using data.prefetch_factor=${DATA_PREFETCH_FACTOR} and data.train_shuffle=${TRAIN_SHUFFLE}"
echo "using validation surface_snapshot=${SURFACE_SNAPSHOT}"
echo "using ddp_broadcast_buffers=${DDP_BROADCAST_BUFFERS} and ddp_timeout_minutes=${DDP_TIMEOUT_MINUTES}"
echo "using ddp_max_data_workers_per_rank=${DDP_MAX_DATA_WORKERS_PER_RANK}"
echo "using data.concurrent_compute=${CONCURRENT_COMPUTE}"
echo "using optimization: learning_rate=${LEARNING_RATE}, scheduler_mode=${SCHEDULER_MODE}, scheduler_target_epochs=${SCHEDULER_TARGET_EPOCHS:-<default>}"
echo "using lr multipliers: lr_multipliers=${LR_MULTIPLIERS}, lr_multiplier_transition=${LR_MULTIPLIER_TRANSITION}"
echo "using curriculum: data_stride=${DATA_STRIDE}, temporal_stride=${TEMPORAL_STRIDE}, steps=${STEPS}, step_transition=${STEP_TRANSITION}, temporal_stride_transition=${TEMPORAL_STRIDE_TRANSITION}, hist=${HIST}, grad-detach=${GRADIENT_DETACH_INTERVAL}"
echo "using data location: LLC face=${LLC_FACE}, i=[${LLC_I_START}:${LLC_I_END}), j=[${LLC_J_START}:${LLC_J_END})"
echo "using padding: pad=${PAD}, num_halo=${NUM_HALO}, num_sponge=${NUM_SPONGE}"
if [[ -n "${DATA_LOCATION_OVERRIDE}" ]]; then
  echo "overriding data.data_location=${DATA_LOCATION_OVERRIDE}"
fi

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

EXPERIMENT_ARGS=(--experiment.name "${EXPERIMENT_NAME}")
echo "overriding experiment.name=${EXPERIMENT_NAME}"
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
  --lr_multipliers "${LR_MULTIPLIERS}"
  --data.hist "${HIST}"
)

DATA_OVERRIDE_ARGS=()
if [[ -n "${DATA_LOCATION_OVERRIDE}" ]]; then
  DATA_OVERRIDE_ARGS+=(--data.data_location "${DATA_LOCATION_OVERRIDE}")
fi

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

LR_MULTIPLIER_TRANSITION_COMPACT="$(echo "${LR_MULTIPLIER_TRANSITION}" | tr -d '[:space:]')"
if [[ -n "${LR_MULTIPLIER_TRANSITION_COMPACT}" && "${LR_MULTIPLIER_TRANSITION_COMPACT}" != "[]" ]]; then
  CURRICULUM_ARGS+=(--lr_multiplier_transition "${LR_MULTIPLIER_TRANSITION}")
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

"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node="${GPUS}" \
  -m ocean_emulators.train configs/samudra_llc/train_normal_2.yaml \
  --save_freq "${SAVE_FREQ}" \
  --epochs "${EPOCHS}" \
  --emergency_checkpoint_interval_minutes "${EMERGENCY_CHECKPOINT_INTERVAL_MINUTES}" \
  "${OPTIM_ARGS[@]}" \
  "${CURRICULUM_ARGS[@]}" \
  --model.pad "${PAD}" \
  --model.num_halo "${NUM_HALO}" \
  --model.num_sponge "${NUM_SPONGE}" \
  --model.gradient_detach_interval "${GRADIENT_DETACH_INTERVAL}" \
  --gradient_accumulation_steps 4 \
  --ddp_bucket_cap_mb 25 \
  --ddp_use_no_sync_for_accumulation true \
  --ddp_broadcast_buffers "${DDP_BROADCAST_BUFFERS}" \
  --ddp_timeout_minutes "${DDP_TIMEOUT_MINUTES}" \
  --ddp_max_data_workers_per_rank "${DDP_MAX_DATA_WORKERS_PER_RANK}" \
  --surface_snapshot "${SURFACE_SNAPSHOT}" \
  --data.num_workers "${DATA_NUM_WORKERS}" \
  --data.prefetch_factor "${DATA_PREFETCH_FACTOR}" \
  --data.train_shuffle "${TRAIN_SHUFFLE}" \
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
