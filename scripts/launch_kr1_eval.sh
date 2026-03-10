#!/bin/bash
# Launch the KR1 multi-scale FOMO eval run on Torch (NYU HPC).
#
# This script submits an eval job for issue #616:
#   Evaluate a model trained on 1/4, 1/2, and 1 degree inputs
#   via long rollout inference.
#
# NOTE: The eval code runs inference on the *first* data source only.
# To evaluate all three scales, run three separate evals by overriding
# CONFIG to point at configs with different source orderings, or by
# reordering sources in the eval config.
#
# Prerequisites:
#   1. Container rebuilt with updated configs.
#   2. Data available on torch at /scratch/$USER/data/:
#        om4_quarterdeg_v2/  (OM4.zarr, OM4_means.zarr, OM4_stds.zarr)
#        om4_halfdeg_v4/     (OM4.zarr, OM4_means.zarr, OM4_stds.zarr)
#        om4_onedeg_v3/      (OM4.zarr, OM4_means.zarr, OM4_stds.zarr)
#   3. A trained checkpoint.
#
# Usage:
#   export CONTAINER_HASH=<sha>
#   export TARGET_CHECKPOINT=/scratch/am16581/runs/<run>/saved_nets/ckpt.pt
#   bash scripts/launch_kr1_eval.sh
set -euo pipefail

# ── Container ──
if [[ -z "${CONTAINER_HASH:-}" && -z "${CONTAINER_TAG:-}" && -z "${IMAGE_REF:-}" ]]; then
  echo "ERROR: Set one of CONTAINER_HASH, CONTAINER_TAG, or IMAGE_REF." >&2
  echo "Example: export CONTAINER_HASH=\$(git rev-parse HEAD)" >&2
  exit 1
fi

# ── Config (baked into the container) ──
export CONFIG=configs/fomo_om4/eval_multiscale.yaml

# ── Run name ──
export NAME_SUFFIX=kr1_fomo_multiscale_eval_v2

# ── Target checkpoint to evaluate ──
export TARGET_CHECKPOINT="${TARGET_CHECKPOINT:-/scratch/am16581/runs/2026-03-06-kr1_fomo_multiscale_v1_1/saved_nets/ckpt.pt}"

# ── Data root: parent dir containing all three resolution subdirectories ──
export DATA_ROOT="${DATA_ROOT:-/scratch/jr7309/data}"

# ── Output base ──
export OUTPUT_BASE="${OUTPUT_BASE:-/scratch/${USER}/runs}"

# ── W&B ──
export WANDB_MODE="${WANDB_MODE:-${WANDB_API_KEY:+online}}"
WANDB_MODE="${WANDB_MODE:-disabled}"

# ── NCCL workarounds for RTX6000 nodes ──
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# ── Extra CLI overrides ──
export ARGS="--model.processor.core_block.pointwise_linear=false"

echo "=== KR1 Multi-Scale FOMO Eval ==="
echo "Config:         ${CONFIG}"
echo "Name suffix:    ${NAME_SUFFIX}"
echo "Data root:      ${DATA_ROOT}"
echo "Output base:    ${OUTPUT_BASE}"
echo "Checkpoint:     ${TARGET_CHECKPOINT}"
echo "W&B mode:       ${WANDB_MODE}"
echo "Container:      ${IMAGE_REF:-${CONTAINER_TAG:-25.11-${CONTAINER_HASH:-???}}}"
echo "ARGS:           ${ARGS}"
echo ""

# ── Submit ──
# 1x RTX6000 for inference.
sbatch \
  --account=torch_pr_347_courant \
  --partition=rtx6000_lzanna \
  --nodes=1 \
  --ntasks-per-node=1 \
  --cpus-per-task=8 \
  --mem=128G \
  --gres=gpu:rtx6000:1 \
  --time=04:00:00 \
  --job-name=kr1-fomo-eval \
  scripts/slurm_apptainer_eval.sbatch
