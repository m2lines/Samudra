#!/bin/bash
# Launch the KR1 (part 2) single-scale 1/2° rollout eval on Torch (NYU HPC).
#
# This evaluates the multi-scale FOMO model trained in part 1 on 1/2°
# initial conditions. The model was trained on 1°, 1/2°, and 1/4° data
# (train_schedule: match). Eval runs inference on the first data source
# in the config, which we set to halfdeg.
#
# Prerequisites:
#   1. Container image available (default: 25.11-latest).
#   2. Data on torch at /scratch/$USER/data/:
#        om4_onedeg_v3/    (required so model builds with right max grid)
#        om4_halfdeg_v4/   (evaluated)
#        om4_quarterdeg_v2/ (required so model builds with right max grid)
#   3. Checkpoint from part 1: v43 EMA weights at epoch 19.
#   4. WANDB_API_KEY set for online tracking.
#
# Usage:
#   export WANDB_API_KEY=<key>
#   export GHCR_USERNAME=<user>   # if private image
#   export GHCR_TOKEN=<token>     # if private image
#   export CONTAINER_HASH=<sha>   # optional; default=latest
#   bash scripts/launch_kr1_eval_halfdeg.sh
#
# Output goes to /scratch/$USER/runs/<YYYY-MM-DD>-kr1_fomo_multiscale_halfdeg_eval_v1/
set -euo pipefail

# ── Container ──
# Default to latest (eval is source-insensitive; no train-code changes needed).
export CONTAINER_TAG="${CONTAINER_TAG:-25.11-latest}"

# ── Config (baked into the container) ──
export CONFIG=configs/fomo_om4/eval_multiscale_halfdeg.yaml

# ── Run name ──
export NAME_SUFFIX=kr1_fomo_multiscale_halfdeg_eval_v1

# ── Data root ──
export DATA_ROOT="${DATA_ROOT:-/scratch/am16581/data}"

# ── Output base ──
export OUTPUT_BASE="${OUTPUT_BASE:-/scratch/${USER}/runs}"

# ── Checkpoint: v43 EMA weights at epoch 19 (best available from part 1) ──
export CKPT_PATH="${CKPT_PATH:-/scratch/am16581/runs/2026-04-21-kr1_fomo_multiscale_v43/saved_nets/ema_ckpt.pt}"

# ── W&B ──
export WANDB_MODE="${WANDB_MODE:-${WANDB_API_KEY:+online}}"
WANDB_MODE="${WANDB_MODE:-disabled}"

# ── Extra CLI overrides ──
# experiment.data_root must be set (no default for it in ExperimentConfig).
export ARGS="--experiment.data_root=${DATA_ROOT}"

echo "=== KR1 Part 2: 1/2° Eval Rollout ==="
echo "Config:       ${CONFIG}"
echo "Name suffix:  ${NAME_SUFFIX}"
echo "Data root:    ${DATA_ROOT}"
echo "Output base:  ${OUTPUT_BASE}"
echo "Checkpoint:   ${CKPT_PATH}"
echo "W&B mode:     ${WANDB_MODE}"
echo "Container:    ${IMAGE_REF:-${CONTAINER_TAG:-25.11-${CONTAINER_HASH:-???}}}"
echo "ARGS:         ${ARGS}"
echo ""

# ── Submit ──
# Single RTX6000 GPU, single node. Eval is not distributed.
sbatch \
  --account=torch_pr_347_courant \
  --nodes=1 \
  --ntasks-per-node=1 \
  --cpus-per-task=4 \
  --mem=64G \
  --gres=gpu:rtx6000:1 \
  --time=04:00:00 \
  --job-name=kr1-eval-halfdeg \
  scripts/slurm_apptainer_eval.sbatch
