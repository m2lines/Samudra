#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --job-name=2026-06-30:samudra_llc:rb-1-TIMING-QUAD-AGULHAS_upscale_factor-1_(ch_width:192,288,384,384)
#SBATCH -x node4100,node3401
#SBATCH -N 1
#SBATCH --mem=254GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH -G h200:1
#SBATCH --time=4:30:00
#SBATCH -o /orcd/home/002/codycruz/Ocean_Emulator/logs/%x-%j.out
#SBATCH -e /orcd/home/002/codycruz/Ocean_Emulator/logs/%x-%j.out

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"
echo "SLURMD_NODENAME=${SLURMD_NODENAME:-<unset>}"
echo "hostname=$(hostname)"

module load miniforge/24.3.0-0
module load cuda/13.1.0

cd /orcd/home/002/codycruz/Ocean_Emulator

PROJECT_SITE_PACKAGES="/orcd/home/002/codycruz/Ocean_Emulator/.venv/lib/python3.11/site-packages"
export PYTHONPATH="/orcd/home/002/codycruz/Ocean_Emulator/src:${PROJECT_SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="${PYTHON_BIN:-/orcd/home/002/codycruz/Ocean_Emulator/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: expected Python at ${PYTHON_BIN}, not executable." >&2
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export NCCL_DEBUG=WARN

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: expected Python 3.11 venv at ${PYTHON_BIN}, but it is not executable." >&2
  exit 1
fi
echo "Using PYTHON_BIN=${PYTHON_BIN}"
"${PYTHON_BIN}" -c 'import sys, torch; print("python", sys.version.split()[0]); print("torch", torch.__version__, "cuda", torch.cuda.is_available())' \
  || { echo "ERROR: venv torch import failed; check environment." >&2; exit 1; }

# ---- timing knobs ----
READ_ITERS="${READ_ITERS:-60}"
GPU_ITERS="${GPU_ITERS:-60}"
STEP_ITERS="${STEP_ITERS:-200}"
WARMUP="${WARMUP:-8}"
CADENCE="${CADENCE:-50}"
READ_THREADS="${READ_THREADS:-1,2,4,6,8,10}"

# ---- data location (match your training run) ----
BATCH_SIZE="${BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
LLC_FACE="${LLC_FACE:-1}"
LLC_I_START="${LLC_I_START:-2160}"
LLC_I_END="${LLC_I_END:-3600}"
LLC_J_START="${LLC_J_START:-0}"
LLC_J_END="${LLC_J_END:-1440}"
DATA_STRIDE="${DATA_STRIDE:-[1]}"
TEMPORAL_STRIDE="${TEMPORAL_STRIDE:-1}"
HIST="${HIST:-0}"
DATA_NUM_WORKERS="${DATA_NUM_WORKERS:-6}"
DATA_PREFETCH_FACTOR="${DATA_PREFETCH_FACTOR:-6}"
BLOSC_THREADS="${BLOSC_THREADS:-1}"
export OCEAN_BLOSC_THREADS="${OCEAN_BLOSC_THREADS:-${BLOSC_THREADS}}"
PIN_MEM="${PIN_MEM:-true}"

# ---- replay knobs (must be self-consistent for buffer init) ----
REPLAY_ENABLED="${REPLAY_ENABLED:-true}"
REPLAY_BUFFER_SIZE="${REPLAY_BUFFER_SIZE:-32}"
REPLAY_REFRESH_EVERY_N_MICROBATCHES="${REPLAY_REFRESH_EVERY_N_MICROBATCHES:-8}"
REPLAY_REFRESH_EVERY_N_MICROBATCHES_TRANSITION="${REPLAY_REFRESH_EVERY_N_MICROBATCHES_TRANSITION:-[]}"
REPLAY_STEPS_PER_EPOCH="${REPLAY_STEPS_PER_EPOCH:-8760}"
REPLAY_MAX_LEAD_STEPS="${REPLAY_MAX_LEAD_STEPS:-[4, 5, 6, 7, 8, 9, 10, 11, 12, 13]}"
REPLAY_MAX_LEAD_TRANSITION="${REPLAY_MAX_LEAD_TRANSITION:-[6, 11, 16, 21, 26, 31, 36, 41, 46]}"

echo "======== replay timing diagnostic ========"
echo "read_iters=${READ_ITERS} gpu_iters=${GPU_ITERS} step_iters=${STEP_ITERS} warmup=${WARMUP}"
echo "read_threads=${READ_THREADS} batch_size=${BATCH_SIZE} gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} effective_batch_size=$((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "data face=${LLC_FACE} i=[${LLC_I_START}:${LLC_I_END}) j=[${LLC_J_START}:${LLC_J_END})"
echo "workers=${DATA_NUM_WORKERS} prefetch=${DATA_PREFETCH_FACTOR} blosc_threads=${OCEAN_BLOSC_THREADS} pin_mem=${PIN_MEM}"
echo "replay refresh_every_n_microbatches=${REPLAY_REFRESH_EVERY_N_MICROBATCHES}, refresh_every_n_microbatches_transition=${REPLAY_REFRESH_EVERY_N_MICROBATCHES_TRANSITION}"

REPLAY_REFRESH_ARGS=()
REPLAY_REFRESH_TRANSITION_COMPACT="$(echo "${REPLAY_REFRESH_EVERY_N_MICROBATCHES_TRANSITION}" | tr -d '[:space:]')"
if [[ -n "${REPLAY_REFRESH_TRANSITION_COMPACT}" && "${REPLAY_REFRESH_TRANSITION_COMPACT}" != "[]" ]]; then
  REPLAY_REFRESH_ARGS+=(--replay.refresh_every_n_microbatches_transition "${REPLAY_REFRESH_EVERY_N_MICROBATCHES_TRANSITION}")
fi

"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=1 \
  scripts/measure_replay_timing.py configs/samudra_llc/train_replay_testing_3.yaml \
  --read-iters "${READ_ITERS}" \
  --gpu-iters "${GPU_ITERS}" \
  --step-iters "${STEP_ITERS}" \
  --warmup "${WARMUP}" \
  --cadence "${CADENCE}" \
  --read-threads "${READ_THREADS}" \
  -- \
  --batch_size "${BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --data_stride "${DATA_STRIDE}" \
  --temporal_stride "${TEMPORAL_STRIDE}" \
  --data.hist "${HIST}" \
  --replay.enabled "${REPLAY_ENABLED}" \
  --replay.buffer_size "${REPLAY_BUFFER_SIZE}" \
  --replay.refresh_every_n_microbatches "${REPLAY_REFRESH_EVERY_N_MICROBATCHES}" \
  "${REPLAY_REFRESH_ARGS[@]}" \
  --replay.steps_per_epoch "${REPLAY_STEPS_PER_EPOCH}" \
  --replay.max_lead_steps "${REPLAY_MAX_LEAD_STEPS}" \
  --replay.max_lead_transition "${REPLAY_MAX_LEAD_TRANSITION}" \
  --data.num_workers "${DATA_NUM_WORKERS}" \
  --data.prefetch_factor "${DATA_PREFETCH_FACTOR}" \
  --pin_mem "${PIN_MEM}" \
  --data.llc_face "${LLC_FACE}" \
  --data.llc_i_start "${LLC_I_START}" \
  --data.llc_i_end "${LLC_I_END}" \
  --data.llc_j_start "${LLC_J_START}" \
  --data.llc_j_end "${LLC_J_END}" \
  --experiment.data_root "/orcd/data/abodner/"

exit $?
