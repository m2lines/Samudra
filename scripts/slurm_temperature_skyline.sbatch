#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=temperature-skyline
#SBATCH --ntasks=1

set -euo pipefail
: "${SKYLINE_ROOT:?}" "${SKYLINE_COMMIT:?}" "${SKYLINE_ARCHIVE_SHA256:?}"
: "${SKYLINE_CONFIG:?}" "${SKYLINE_OUTPUT:?}" "${SIF_PATH:?}"
export APPTAINER_CACHEDIR=/scratch/jr7309/apptainer-cache
export SINGULARITY_CACHEDIR=/scratch/jr7309/singularity-cache
export APPTAINER_TMPDIR="${SLURM_TMPDIR:-/tmp}"
export WANDB_DIR="${SKYLINE_OUTPUT}/wandb"
export WANDB_CACHE_DIR="${SKYLINE_ROOT}/wandb-cache"
export WANDB_CONFIG_DIR="${SKYLINE_ROOT}/wandb-config"
export PYTHONUNBUFFERED=1
mkdir -p "${SKYLINE_OUTPUT}" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"
ARCHIVE="${SKYLINE_ROOT}/source-${SKYLINE_COMMIT}.tar"
actual=$(sha256sum "$ARCHIVE" | cut -d ' ' -f 1)
test "$actual" = "$SKYLINE_ARCHIVE_SHA256"
# Extract a verified immutable commit archive into node-local storage for each job.
CODE_DIR=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/skyline-code.XXXXXX")
trap 'rm -rf "$CODE_DIR"' EXIT
tar -xf "$ARCHIVE" -C "$CODE_DIR"
gpu_flags=()
if [[ "${SKYLINE_MODE:-train}" == train ]]; then
    gpu_flags=(--nv)
    nvidia-smi
fi
printf 'W&B mode: %s\n' "${WANDB_MODE:-online}"
printf 'Source commit: %s\n' "$SKYLINE_COMMIT"
extra=()
if [[ -n "${SKYLINE_AUDIT_DATES:-}" ]]; then extra=(--audit-dates "$SKYLINE_AUDIT_DATES"); fi
apptainer exec "${gpu_flags[@]}" \
    --bind "${CODE_DIR}:/opt/skyline-source:ro,/scratch:/scratch" \
    "$SIF_PATH" /usr/bin/env \
    PYTHONPATH=/opt/skyline-source/src \
    TORCH_CUDNN_V8_API_DISABLED=1 \
    /workspace/.venv/bin/python -m samudra.skyline.run "${SKYLINE_MODE:-train}" \
    --config "$SKYLINE_CONFIG" --output "$SKYLINE_OUTPUT" "${extra[@]}"
