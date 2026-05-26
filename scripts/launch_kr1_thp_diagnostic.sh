#!/bin/bash

# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

# v25: Diagnose Transparent Huge Page (THP) defrag stalls.
#
# Hypothesis (from Jesse's findings on different hardware):
#   With multiple DataLoader workers per GPU, the default THP defrag setting
#   (`madvise`) causes the kernel to synchronously defragment memory when
#   mimalloc (used by PyTorch) requests huge pages via madvise(MADV_HUGEPAGE).
#   This manifests as a multi-minute training stall after ~8-24 steps as
#   workers warm up and memory pressure grows.
#
# This run collects:
#   1. THP kernel settings at job start
#   2. Periodic /proc/vmstat snapshots (THP + compaction counters)
#   3. strace on one DataLoader worker to capture madvise() calls
#   4. py-spy watchdog for stack context during stalls
#   5. Standard experiment.log for iter_time correlation
#
# The TensorMap.to(device) fix is included (already in train.py on this branch).
#
# Usage:
#   export CONTAINER_HASH=<sha>
#   bash scripts/launch_kr1_thp_diagnostic.sh
set -euo pipefail

# ── Container ──
if [[ -z "${CONTAINER_HASH:-}" && -z "${CONTAINER_TAG:-}" && -z "${IMAGE_REF:-}" ]]; then
  echo "ERROR: Set one of CONTAINER_HASH, CONTAINER_TAG, or IMAGE_REF." >&2
  exit 1
fi

# ── Config ──
export CONFIG=configs/fomo_om4/train_multiscale.yaml

# ── Run name ──
export NAME_SUFFIX=kr1_fomo_multiscale_v25

# ── Data root ──
export DATA_ROOT="${DATA_ROOT:-/scratch/am16581/data}"

# ── Output base ──
export OUTPUT_BASE="${OUTPUT_BASE:-/scratch/${USER}/runs}"

# ── W&B: disabled for diagnostic run ──
export WANDB_MODE=disabled

# ── No preemption ──
export PREEMPTIBLE=0

# ── Checkpoint every 100 batches ──
export CHECKPOINT_BATCH_INTERVAL=100

# ── NCCL workarounds for RTX6000 nodes ──
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export UCX_TLS=tcp,self,sm
export UCX_NET_DEVICES=all
export NCCL_NET=Socket

# ── Stage data to node-local NVMe ──
export STAGE_DATA=0
export STAGE_DST=/state/partition1/data
export STAGE_SOURCES="om4_quarterdeg_v2 om4_halfdeg_v4 om4_onedeg_v3"
export STAGE_TIME_START="1975-01-03"
export STAGE_TIME_END="2014-10-05"

# ── py-spy watchdog: periodic stack dumps to catch stalls ──
export PYSPY_WATCHDOG=1
export PYSPY_INTERVAL=30  # More frequent than usual to catch THP stalls

# ── No nsys this run — focus on syscall/vmstat diagnostics ──
export NSYS_PROFILE=0

# ── THP diagnostic monitoring (picked up by sbatch script) ──
export THP_MONITOR=1
export THP_MONITOR_INTERVAL=5  # Sample /proc/vmstat every 5 seconds

# ── strace madvise on one DataLoader worker ──
export STRACE_MADVISE=1

# ── CLI overrides ──
export ARGS="--data.num_workers=8 --data.concurrent_compute=true"

echo "=== v25: THP Defrag Diagnostic Run ==="
echo "Config:         ${CONFIG}"
echo "Name suffix:    ${NAME_SUFFIX}"
echo "Data root:      ${DATA_ROOT}"
echo "Output base:    ${OUTPUT_BASE}"
echo "W&B mode:       ${WANDB_MODE}"
echo "Container:      ${IMAGE_REF:-${CONTAINER_TAG:-25.11-${CONTAINER_HASH:-???}}}"
echo "THP monitor:    interval=${THP_MONITOR_INTERVAL}s"
echo "strace madvise: ${STRACE_MADVISE}"
echo ""

sbatch \
  --account=torch_pr_347_courant \
  --nodes=1 \
  --ntasks-per-node=1 \
  --cpus-per-task=128 \
  --mem=1400G \
  --gres=gpu:rtx6000:8 \
  --time=4:00:00 \
  --job-name=kr1-thp-diag \
  scripts/slurm_apptainer_train.sbatch
