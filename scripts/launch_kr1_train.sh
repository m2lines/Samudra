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
export NAME_SUFFIX=kr1_fomo_multiscale

# ── Data root: parent dir containing all three resolution subdirectories ──
export DATA_ROOT="${DATA_ROOT:-/scratch/jr7309/data}"

# ── Output base ──
export OUTPUT_BASE="${OUTPUT_BASE:-/scratch/${USER}/runs}"

# ── W&B ──
export WANDB_MODE="${WANDB_MODE:-${WANDB_API_KEY:+online}}"
WANDB_MODE="${WANDB_MODE:-disabled}"

# ── Extra CLI overrides ──
# The baked-in config has the decoder and data sources already configured.
# We just pass the W&B project and any batch size tweaks here.
export ARGS="--batch_size=2"

echo "=== KR1 Multi-Scale FOMO Training ==="
echo "Config:         ${CONFIG}"
echo "Name suffix:    ${NAME_SUFFIX}"
echo "Data root:      ${DATA_ROOT}"
echo "Output base:    ${OUTPUT_BASE}"
echo "W&B mode:       ${WANDB_MODE}"
echo "Container:      ${IMAGE_REF:-${CONTAINER_TAG:-25.11-${CONTAINER_HASH:-???}}}"
echo "ARGS:           ${ARGS}"
echo ""

# ── Submit ──
# 8x RTX6000, full node on torch.
# Time: 72 hours for a long training run (70 epochs x 3 scales).
sbatch \
  --account=torch_pr_347_courant \
  --nodes=1 \
  --ntasks-per-node=1 \
  --cpus-per-task=128 \
  --mem=1400G \
  --gres=gpu:rtx6000:8 \
  --time=72:00:00 \
  --job-name=kr1-fomo \
  scripts/slurm_apptainer_train.sbatch
