#!/bin/bash
# Launch the KR1 multi-scale FOMO training run on Torch (NYU HPC).
#
# This script submits the training job for issue #616:
#   A model trained on 1/4, 1/2, and 1 degree inputs can produce outputs
#   at those resolutions after a 10-year rollout.
#
# Prerequisites:
#   1. Container rebuilt with updated configs (see below).
#   2. Data available on torch at /scratch/$USER/data/:
#        om4_quarterdeg_v2/  (OM4.zarr, OM4_means.zarr, OM4_stds.zarr)
#        om4_halfdeg_v4/     (OM4.zarr, OM4_means.zarr, OM4_stds.zarr)
#        om4_onedeg_v3/      (OM4.zarr, OM4_means.zarr, OM4_stds.zarr)
#   3. WANDB_API_KEY set if you want online logging.
#
# Usage:
#   # First, rebuild the container from the kr1 branch:
#   gh workflow run "Container PhysicsNeMo 25.11" --ref u/alxmrs/experiments/kr1
#   # Wait for it to finish, then get the SHA:
#   CONTAINER_HASH=$(git rev-parse HEAD)
#
#   # Then launch:
#   export CONTAINER_HASH=<sha>
#   bash scripts/launch_kr1_train.sh
#
# The run output (checkpoints, logs) will be at:
#   /scratch/$USER/runs/<YYYY-MM-DD>-kr1_fomo_multiscale/
#
# Checkpoints saved:
#   saved_nets/latest_ckpt.pt    — every epoch
#   saved_nets/ema_ckpt.pt       — EMA weights (every epoch)
#   saved_nets/ckpt_epoch_N.pt   — every 5 epochs
#   saved_nets/best_val_ckpt.pt  — best validation loss
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
export NAME_SUFFIX=kr1_fomo_multiscale_v45

# ── Data root: parent dir containing all three resolution subdirectories ──
export DATA_ROOT="${DATA_ROOT:-/scratch/am16581/data}"

# ── Output base ──
export OUTPUT_BASE="${OUTPUT_BASE:-/scratch/${USER}/runs}"

# ── W&B ──
export WANDB_MODE="${WANDB_MODE:-${WANDB_API_KEY:+online}}"
WANDB_MODE="${WANDB_MODE:-disabled}"

# ─ Use preemptable resources, make the job resumable. ──
export PREEMPTIBLE=0  # Turn off preemption

# ─ Checkpoint every 100 batches, not 250. ──
export CHECKPOINT_BATCH_INTERVAL=100

# ── NCCL workarounds for RTX6000 nodes ──
# P2P and IB cause hangs/segfaults on gr101/gr102; disable them.
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# Allow longer data-loading stalls before NCCL kills the job (default 600s).
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
# UCX (transport layer under NCCL) also tries IB by default and segfaults on
# nodes without proper IB support.  Restrict to TCP + shared memory.
export UCX_TLS=tcp,self,sm
export UCX_NET_DEVICES=all
# Force NCCL to use the built-in socket transport instead of UCX plugin.
# UCX's libucs.so segfaults intermittently on gr101/gr102 during NCCL init.
export NCCL_NET=Socket

# ── Stage data to node-local NVMe to avoid /scratch I/O stalls ──
export STAGE_DATA=0
export STAGE_DST=/state/partition1/data
export STAGE_SOURCES="om4_quarterdeg_v2 om4_halfdeg_v4 om4_onedeg_v3"
export STAGE_TIME_START="1975-01-03"
export STAGE_TIME_END="2014-10-05"

# ── py-spy watchdog: periodic stack dumps to diagnose DataLoader stalls ──
export PYSPY_WATCHDOG=1
export PYSPY_INTERVAL=60

# ── py-spy record: flamegraph profiling at 100hz ──
export PYSPY_RECORD=0
export PYSPY_RECORD_RATE=100
export PYSPY_RECORD_FORMAT=speedscope

# ── nsys profiling ──
# Disabled for v26 — real training run with THP defrag fix confirmed.
export NSYS_PROFILE=0

# ── Extra CLI overrides ──
# The baked-in config has the decoder and data sources already configured.
# We just pass the W&B project and any batch size tweaks here.
export ARGS="--data.loading.num_workers=8 --data.concurrent_compute=true --resume_ckpt_path=/scratch/am16581/runs/2026-04-21-kr1_fomo_multiscale_v43/saved_nets/ckpt.pt"

echo "=== KR1 Multi-Scale FOMO Training ==="
echo "Config:         ${CONFIG}"
echo "Name suffix:    ${NAME_SUFFIX}"
echo "Data root:      ${DATA_ROOT}"
echo "Output base:    ${OUTPUT_BASE}"
echo "W&B mode:       ${WANDB_MODE}"
echo "Container:      ${IMAGE_REF:-${CONTAINER_TAG:-25.11-${CONTAINER_HASH:-???}}}"
echo "ARGS:           ${ARGS}"
echo "PREEMPTIBLE:    ${PREEMPTIBLE}"
echo "CKPT_BATCH_INT: ${CHECKPOINT_BATCH_INTERVAL}"
echo ""

# ── Submit ──
# 8x RTX6000, full node on torch.
# Time: 2h per run. The gpu48 QOS (auto-assigned for >2h) caps GPUs at 2,
# so we stay within the default QOS and rely on PREEMPTIBLE=1 to auto-requeue
# after walltime / preemption and resume from checkpoint.
sbatch \
  --account=torch_pr_347_courant \
  --nodes=1 \
  --ntasks-per-node=1 \
  --cpus-per-task=128 \
  --mem=1400G \
  --gres=gpu:rtx6000:8 \
  --time=24:00:00 \
  --job-name=kr1-fomo \
  scripts/slurm_apptainer_train.sbatch

# Turn off preemption
#  --time=02:00:00 \
#  --requeue \
#  --comment="preemption=yes;requeue=true" \
