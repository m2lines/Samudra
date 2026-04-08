#!/bin/bash
# Profile a KR1 multi-scale FOMO training run with NVIDIA Nsight Systems.
#
# Launches the normal 8-GPU distributed training under `nsys profile` to
# capture CUDA kernels, NCCL collectives, CPU samples, and OS runtime calls.
# The profile is time-limited (default 10 minutes after a warmup delay) so
# nsys shuts down cleanly — do NOT ctrl-C the job.
#
# The resulting .nsys-rep files can be opened in the Nsight Systems GUI or
# uploaded to an analysis machine.  They land in the run directory:
#   /scratch/$USER/runs/<YYYY-MM-DD>-kr1_fomo_nsys_profile/nsys_*.nsys-rep
#
# Usage:
#   export CONTAINER_HASH=<sha>
#   bash scripts/launch_kr1_nsys_profile.sh
#
# Optional overrides:
#   NSYS_DELAY=120      Seconds to wait before profiling starts (default: 120)
#   NSYS_DURATION=600   Seconds of profiling to capture (default: 600)
#   NSYS_DETAIL=1       Set to 1 for detailed python+cuda backtraces (higher overhead)
set -euo pipefail

# ── Container ──
if [[ -z "${CONTAINER_HASH:-}" && -z "${CONTAINER_TAG:-}" && -z "${IMAGE_REF:-}" ]]; then
  echo "ERROR: Set one of CONTAINER_HASH, CONTAINER_TAG, or IMAGE_REF." >&2
  echo "Example: export CONTAINER_HASH=\$(git rev-parse HEAD)" >&2
  exit 1
fi

# ── Config (baked into the container) ──
export CONFIG=configs/fomo_om4/train_multiscale.yaml

# ── Run name ──
export NAME_SUFFIX=kr1_fomo_nsys_profile_v2

# ── Data root ──
export DATA_ROOT="${DATA_ROOT:-/scratch/jr7309/data}"

# ── Output base ──
export OUTPUT_BASE="${OUTPUT_BASE:-/scratch/${USER}/runs}"

# ── W&B disabled for profiling ──
export WANDB_MODE=disabled

# ── No preemption ──
export PREEMPTIBLE=0

# ── NCCL workarounds for RTX6000 nodes ──
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export UCX_TLS=tcp,self,sm
export UCX_NET_DEVICES=all

# ── nsys timing ──
# Delay: skip startup so the profile captures steady-state training.
#        With ~10s/step, 400s ≈ step 40, well before the deadlock zone (~58).
# Duration: 300s captures through the deadlock and leaves headroom before the
#           hardcoded 600s DataLoader timeout kills the process.  nsys must
#           finish and write .nsys-rep files BEFORE that timeout fires.
NSYS_DELAY="${NSYS_DELAY:-400}"
NSYS_DURATION="${NSYS_DURATION:-300}"
NSYS_DETAIL="${NSYS_DETAIL:-0}"

# ── Compute the NAME with date prefix (same logic as sbatch) ──
NAME="$(date +%Y-%m-%d)-${NAME_SUFFIX}"
export NAME

# ── Build the nsys command ──
# Profile output goes into the run directory.
RUN_DIR="${OUTPUT_BASE}/${NAME}"

# Base flags: low-overhead snapshot of CUDA, NCCL, OS runtime, CPU sampling.
NSYS_FLAGS=(
  --trace=cuda,nvtx,osrt,nccl
  --sample=cpu
  --delay="${NSYS_DELAY}"
  --duration="${NSYS_DURATION}"
  --kill=sigterm
  --output="${RUN_DIR}/nsys_%h_%p"
  --force-overwrite=true
  --stats=true
)

# Detailed mode: adds python backtraces on CUDA events and python sampling.
# Higher overhead but shows which python code triggers each CUDA kernel.
if [[ "${NSYS_DETAIL}" == "1" ]]; then
  NSYS_FLAGS+=(
    --cudabacktrace=all
    --python-backtrace=cuda
    --python-sampling=true
  )
fi

NSYS_CMD="nsys profile ${NSYS_FLAGS[*]}"

# ── Extra CLI overrides ──
# Match the real training run's args so the profile is representative.
# Keep concurrent_compute enabled — we WANT to reproduce the deadlock so
# nsys captures the CUDA/NCCL state at that moment.
export ARGS="--data.num_workers=8 --data.concurrent_compute=true"

# ── Walltime: delay + duration + buffer for startup and nsys teardown ──
# Buffer must be generous: nsys needs a clean shutdown to write .nsys-rep
# files.  Training steps take 8-13s each, and data loading can stall for
# 30-60s on resolution switches, so 300s was too tight.
TOTAL_SECS=$(( NSYS_DELAY + NSYS_DURATION + 900 ))
WALLTIME="$(printf '%02d:%02d:%02d' $((TOTAL_SECS/3600)) $(((TOTAL_SECS%3600)/60)) $((TOTAL_SECS%60)))"

echo "=== KR1 Nsight Systems Profiling ==="
echo "Config:        ${CONFIG}"
echo "Name:          ${NAME}"
echo "Data root:     ${DATA_ROOT}"
echo "Output base:   ${OUTPUT_BASE}"
echo "Container:     ${IMAGE_REF:-${CONTAINER_TAG:-25.11-${CONTAINER_HASH:-???}}}"
echo "ARGS:          ${ARGS}"
echo "nsys delay:    ${NSYS_DELAY}s"
echo "nsys duration: ${NSYS_DURATION}s"
echo "nsys detail:   ${NSYS_DETAIL}"
echo "Walltime:      ${WALLTIME}"
echo "Profile out:   ${RUN_DIR}/nsys_*.nsys-rep"
echo ""

# ── Submit ──
# We inject NSYS_CMD as an env var.  The sbatch script will prefix torchrun
# with this command, producing one .nsys-rep per process (torchrun parent +
# 8 GPU workers).
sbatch \
  --account=torch_pr_347_courant \
  --nodes=1 \
  --ntasks-per-node=1 \
  --cpus-per-task=128 \
  --mem=1400G \
  --gres=gpu:rtx6000:8 \
  --time="${WALLTIME}" \
  --job-name=kr1-nsys \
  --export="ALL,NSYS_CMD=${NSYS_CMD}" \
  scripts/slurm_apptainer_nsys_profile.sbatch
