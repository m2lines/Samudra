#!/bin/bash

# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

# Profile CPU memory for a single rank of the KR1 multi-scale FOMO training.
#
# Runs memray on a single GPU (no DDP) to measure per-rank memory footprint.
# This doesn't require a container rebuild since memray is already installed.
#
# Usage:
#   export CONTAINER_HASH=<sha>
#   bash scripts/launch_kr1_profile.sh
set -euo pipefail

# ── Container ──
if [[ -z "${CONTAINER_HASH:-}" && -z "${CONTAINER_TAG:-}" && -z "${IMAGE_REF:-}" ]]; then
  echo "ERROR: Set one of CONTAINER_HASH, CONTAINER_TAG, or IMAGE_REF." >&2
  exit 1
fi

# ── Config (baked into the container) ──
export CONFIG=configs/fomo_om4/train_multiscale.yaml

# ── Run name ──
export NAME_SUFFIX=kr1_fomo_profile_memory_v2

# ── Data root ──
export DATA_ROOT="${DATA_ROOT:-/scratch/jr7309/data}"

# ── Output base ──
export OUTPUT_BASE="${OUTPUT_BASE:-/scratch/${USER}/runs}"

# ── W&B disabled for profiling ──
export WANDB_MODE=disabled

# ── No preemption ──
export PREEMPTIBLE=0

# ── Override: 0 workers (avoid fork OOM), 1 epoch, no inference ──
export ARGS="--batch_size=1 --data.num_workers=0 --epochs=1"

# ── Profile output location (inside the run dir, bind-mounted) ──
PROFILE_OUT="/scratch/${USER}/runs/memprofile_kr1.bin"

echo "=== KR1 Memory Profiling (single GPU, no DDP) ==="
echo "Config:       ${CONFIG}"
echo "Name suffix:  ${NAME_SUFFIX}"
echo "Data root:    ${DATA_ROOT}"
echo "Profile out:  ${PROFILE_OUT}"
echo "ARGS:         ${ARGS}"
echo ""

# ── Submit: single GPU, matching previous successful 1-GPU jobs ──
sbatch \
  --job-name=kr1-profile \
  --export=ALL,PROFILE_OUT="${PROFILE_OUT}" \
  scripts/slurm_apptainer_profile.sbatch
