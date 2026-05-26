#!/bin/bash

# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

# KR1 smoke test: single-scale 1° FOMO training on Torch (NYU HPC).
#
# Why this run:
#   v44 (multi-scale) plateaued at climatology-level val loss with spiky
#   curves. This branch fixes:
#     1. loss: mse (was DynamicLoss with no limit -> spike feedback)
#     2. pred_residuals: true (was false -> trivial 0-output attractor)
#     3. FlashPerceiver 2D Fourier features (was 1D rotary on latents only)
#     4. Per-scale validation aggregator (multi-scale had empty image dict)
#
# Before re-launching multi-scale (which costs 24h x 8 GPUs), we run the
# cheapest possible ablation: same FOMO architecture, same fixes, ONE
# scale only. If onedeg can't beat climatology here, the bug is in the
# model and multi-scale is the wrong knob. If it does, we have a healthy
# baseline curve to compare the next multi-scale run against.
#
# Prerequisites:
#   1. Container available (the train sbatch bind-mounts host src/configs,
#      so source-only changes do NOT need a rebuild; an existing tag works).
#   2. Data on torch at /scratch/$USER/data/om4_onedeg_v3/.
#   3. WANDB_API_KEY set if you want online logging.
#
# Usage:
#   export CONTAINER_HASH=<sha>            # or CONTAINER_TAG=25.11-latest
#   export WANDB_API_KEY=<key>
#   bash scripts/launch_kr1_smoke_onedeg.sh
#
# Output: /scratch/$USER/runs/<YYYY-MM-DD>-kr1_fomo_singlescale_onedeg_v1/
set -euo pipefail

if [[ -z "${CONTAINER_HASH:-}" && -z "${CONTAINER_TAG:-}" && -z "${IMAGE_REF:-}" ]]; then
  echo "ERROR: Set one of CONTAINER_HASH, CONTAINER_TAG, or IMAGE_REF." >&2
  echo "Example: export CONTAINER_TAG=25.11-latest" >&2
  exit 1
fi

# ── Config ──
export CONFIG=configs/fomo_om4/train_singlescale_onedeg.yaml

# ── Run name (bump version on each re-launch) ──
export NAME_SUFFIX=kr1_fomo_singlescale_onedeg_v1

# ── Data root: parent dir containing om4_onedeg_v3/ ──
export DATA_ROOT="${DATA_ROOT:-/scratch/am16581/data}"

# ── Output base ──
export OUTPUT_BASE="${OUTPUT_BASE:-/scratch/${USER}/runs}"

# ── W&B ──
export WANDB_MODE="${WANDB_MODE:-${WANDB_API_KEY:+online}}"
WANDB_MODE="${WANDB_MODE:-disabled}"

# ── NCCL workarounds for RTX6000 nodes (carried over from v45) ──
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export UCX_TLS=tcp,self,sm
export UCX_NET_DEVICES=all
export NCCL_NET=Socket

# ── No data staging: 1° dataset is small and reads cheap from /scratch. ──
export STAGE_DATA=0

# ── No profiling for the smoke run; we want signal speed, not perf data. ──
export NSYS_PROFILE=0
export PYSPY_WATCHDOG=0
export PYSPY_RECORD=0

# ── Extra CLI overrides ──
# Tag the run distinctly so it doesn't merge with the multi-scale runs in
# wandb. No resume — we cannot load v44/v45 checkpoints because the
# FlashPerceiver input projection shape changed (Fourier features add 18
# input channels).
export ARGS="\
  --data.loading.num_workers=8 \
  --data.concurrent_compute=true \
  --experiment.wandb.group=kr1_smoke\
"

echo "=== KR1 smoke: 1° single-scale FOMO ==="
echo "Config:       ${CONFIG}"
echo "Name suffix:  ${NAME_SUFFIX}"
echo "Data root:    ${DATA_ROOT}"
echo "Output base:  ${OUTPUT_BASE}"
echo "W&B mode:     ${WANDB_MODE}"
echo "Container:    ${IMAGE_REF:-${CONTAINER_TAG:-25.11-${CONTAINER_HASH:-???}}}"
echo "ARGS:         ${ARGS}"
echo ""

# ── Submit ──
# 8x RTX6000 gives the fastest signal turnaround for the smoke test, even
# though 1° doesn't saturate them; we're optimizing for time-to-decision.
# 12h walltime is generous: 20 epochs of 1° on 8 GPUs should finish well
# under that and we can kill early if the curve is decisive.
sbatch \
  --account=torch_pr_347_courant \
  --nodes=1 \
  --ntasks-per-node=1 \
  --cpus-per-task=128 \
  --mem=1400G \
  --gres=gpu:rtx6000:8 \
  --time=12:00:00 \
  --job-name=kr1-smoke-onedeg \
  scripts/slurm_apptainer_train.sbatch
