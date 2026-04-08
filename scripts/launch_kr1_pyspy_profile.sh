#!/bin/bash
# Launch a KR1 training run with an embedded py-spy deadlock watchdog.
#
# Submits a single SLURM job that runs training AND a background watchdog.
# When training stalls (no new step for 120s), the watchdog automatically
# dumps py-spy stack traces of all training processes, showing exactly
# which lock/futex each thread is blocked on.
#
# Everything runs inside the same SLURM job — no SSH, no tmux, no manual
# intervention needed.
#
# Usage:
#   export CONTAINER_HASH=<sha>
#   bash scripts/launch_kr1_pyspy_profile.sh
#
# Output:
#   slurm-<JOB_ID>.out                          Training + watchdog logs
#   pyspy_dump_<JOB_ID>_<timestamp>.txt          Stack dumps (in submit dir)

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
export NAME_SUFFIX=kr1_fomo_pyspy_profile_v6

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

# ── Extra CLI overrides ──
# Keep concurrent_compute=true to reproduce the deadlock.
export ARGS="--data.num_workers=8 --data.concurrent_compute=true"

echo "=== KR1 py-spy Deadlock Profiling ==="
echo "Config:         ${CONFIG}"
echo "Name suffix:    ${NAME_SUFFIX}"
echo "Data root:      ${DATA_ROOT}"
echo "Output base:    ${OUTPUT_BASE}"
echo "Container:      ${IMAGE_REF:-${CONTAINER_TAG:-25.11-${CONTAINER_HASH:-???}}}"
echo "ARGS:           ${ARGS}"
echo ""

# ── Submit combined training + watchdog job ──
TRAIN_JOB_ID=$(sbatch --parsable \
  --account=torch_pr_347_courant \
  --nodes=1 \
  --ntasks-per-node=1 \
  --cpus-per-task=128 \
  --mem=1400G \
  --gres=gpu:rtx6000:8 \
  --time=02:00:00 \
  --job-name=kr1-pyspy \
  scripts/slurm_apptainer_pyspy_train.sbatch)

echo "Job submitted: ${TRAIN_JOB_ID}"
echo ""
echo "Monitor with:"
echo "  tail -f slurm-${TRAIN_JOB_ID}.out              # training + watchdog logs"
echo "  ls pyspy_dump_${TRAIN_JOB_ID}_*.txt             # stack dumps (after stall)"
