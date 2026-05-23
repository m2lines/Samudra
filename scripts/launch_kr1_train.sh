#!/bin/bash

# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

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
#   # The slurm_apptainer_train.sbatch harness bind-mounts host src/ and
#   # configs/ into the container, so source-only branch changes do NOT
#   # require a container rebuild — any existing tag works.
#   export CONTAINER_TAG=25.11-latest         # or CONTAINER_HASH=<sha>
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
export CONFIG=configs/fomo_om4/train_multiscale_v51.yaml

# ── Run name ──
# v46: fresh start on this branch (kr1-v2). Cannot resume from v43/v44/v45
# because the FlashPerceiver input projection now takes +18 channels for the
# 2D Fourier features fix; checkpoint shapes are incompatible. Other deltas:
# loss=mse (was DynamicLoss), pred_residuals=true (was false), per-scale
# validation snapshots, single-step warmup via steps=[1,2] step_transition=[10].
# v49: fresh start again — hist 0→1 changes input channel count
# (FlashPerceiver projection shape), so v48 checkpoints don't load. Deltas:
# hist=1 (v48 collapsed to climatology w/ hist=0), pred_residuals=true (v48
# false), steps=[1,2,4] step_transition=[20,45] (extended 1-step phase + 4-step
# cap), lr=0.0003 (halved to damp v47 sawtooth risk).
# v50: v49 NaN'd on iter 1 of the first 2-step epoch (curriculum transition at
# epoch 20). Reverting pred_residuals to false; std_ratio diagnostic now wired
# into PerScale validation path. Keeping hist=1, steps=[1,2,4], transitions
# [20,45]; LR back to v48's 0.0006.
# v51: v50 cleared the 1→2-step transition cleanly (no NaN), then DataLoader
# hung 30 min at epoch 20 iter 356 — classic THP-defrag stall (kernel defrag
# is back to `madvise` on gr101/gr102; sysadmin's v26 `never` setting was
# reverted). Model delta: decoder.context_patches 3 → null (drop, no longer
# needed with v50's loss profile). Everything else identical to v50.
export NAME_SUFFIX=kr1_fomo_multiscale_v51
# Lock NAME at submit time so requeues + chain jobs use the same run dir.
# Override via NAME=... in env to resume into a specific existing dir.
export NAME="${NAME:-$(date +%Y-%m-%d)-${NAME_SUFFIX}}"

# ── Data root: parent dir containing all three resolution subdirectories ──
export DATA_ROOT="${DATA_ROOT:-/scratch/am16581/data}"

# ── Output base ──
export OUTPUT_BASE="${OUTPUT_BASE:-/scratch/${USER}/runs}"

# ── W&B ──
export WANDB_MODE="${WANDB_MODE:-${WANDB_API_KEY:+online}}"
WANDB_MODE="${WANDB_MODE:-disabled}"

# ─ Preemption ──
# v49: HPC@ granted exemption from the post-2h ≥50% GPU-util preemption rule,
# so the job will run uninterrupted within walltime. Use a normal (non-
# preemptible) launch per docs/torch.md.
# Override via PREEMPTIBLE=1 in env to resume from latest_ckpt.pt in an
# existing NAME dir (chained 48h jobs to reach 70 epochs).
export PREEMPTIBLE="${PREEMPTIBLE:-0}"

# ─ Checkpoint every 100 batches, not 250. ──
# Kept for resilience against unexpected failures (node reboot, OOM, etc.),
# even though we're no longer relying on preempt-resume.
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
# v46: NO resume — see NAME_SUFFIX comment above. Fresh weights only.
export ARGS="--data.loading.num_workers=8 --data.concurrent_compute=true --experiment.wandb.group=kr1_v51"

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
# v49: standard non-preemptible 8x RTX6000 full-node launch per docs/torch.md.
# HPC@ has waived the post-2h ≥50% GPU-util preemption rule, so the job runs
# uninterrupted within walltime. No --partition (let SLURM place), no
# --requeue/--comment (no preempt-resume needed).
# `EXTRA_SBATCH_ARGS` is an optional escape hatch for callers that want to
# splice in extra sbatch flags (e.g. `--dependency=afterany:<jobid>` for
# job chaining). Empty by default.
sbatch \
  --account=torch_pr_347_courant \
  --nodes=1 \
  --ntasks-per-node=1 \
  --cpus-per-task=128 \
  --mem=1400G \
  --gres=gpu:rtx6000:8 \
  --time="${WALLTIME:-24:00:00}" \
  --job-name=kr1-fomo \
  ${EXTRA_SBATCH_ARGS:-} \
  scripts/slurm_apptainer_train.sbatch
