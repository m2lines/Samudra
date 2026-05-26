#!/bin/bash

# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

# Slurm job to compact all FOMO multiscale Zarr datasets using Apptainer.
#
# This reads the uncompacted zarr data from SRC_ROOT and writes compacted
# versions (variables merged along a `lev` dimension) to DST_ROOT.
#
# Usage:
#   export SRC_ROOT=/scratch/jr7309/data
#   export DST_ROOT=/scratch/$USER/data
#   sbatch scripts/slurm_compact_data.sbatch
#
# Optional:
#   export SOURCES="om4_onedeg_v3/OM4.zarr om4_onedeg_v3/OM4_means.zarr"  # subset
#   export SIF_PATH=/path/to/container.sif  # specific container image

#SBATCH --job-name=oe-compact
#SBATCH --output=slurm-compact-%j.out
#SBATCH --account=torch_pr_347_courant
#SBATCH --partition=l40s_courant
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=12:00:00

set -euo pipefail

SRC_ROOT="${SRC_ROOT:-/scratch/jr7309/data}"
DST_ROOT="${DST_ROOT:-/scratch/${USER}/data}"

if [[ ! -d "${SRC_ROOT}" ]]; then
  echo "ERROR: SRC_ROOT does not exist: ${SRC_ROOT}" >&2
  exit 1
fi

mkdir -p "${DST_ROOT}"

SOURCES="${SOURCES:-}"
SOURCES_ARGS=""
if [[ -n "${SOURCES}" ]]; then
  SOURCES_ARGS="--sources ${SOURCES}"
fi

export PYTHONUNBUFFERED=1

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"

# ── Container setup (reuse the same Apptainer workflow as training) ──
if command -v apptainer >/dev/null 2>&1; then
  APPTAINER_BIN=apptainer
elif command -v singularity >/dev/null 2>&1; then
  APPTAINER_BIN=singularity
else
  echo "Neither apptainer nor singularity is available on PATH." >&2
  exit 1
fi

# Find a SIF to use: explicit SIF_PATH, or pick the latest cached image.
SIF_PATH="${SIF_PATH:-}"
if [[ -z "${SIF_PATH}" ]]; then
  SIF_DIR="${SIF_DIR:-${REPO_DIR}/.apptainer-images}"
  if [[ -d "${SIF_DIR}" ]]; then
    SIF_PATH="$(ls -t "${SIF_DIR}"/*.sif 2>/dev/null | head -1)"
  fi
fi
if [[ -z "${SIF_PATH}" || ! -f "${SIF_PATH}" ]]; then
  echo "ERROR: No container SIF found. Set SIF_PATH or ensure .apptainer-images/ has a .sif file." >&2
  exit 2
fi

echo "============================================"
echo "Compact Zarr Data"
echo "============================================"
echo "SRC_ROOT:           ${SRC_ROOT}"
echo "DST_ROOT:           ${DST_ROOT}"
echo "SOURCES:            ${SOURCES:-all multiscale}"
echo "Node:               $(hostname)"
echo "CPUs:               ${SLURM_CPUS_PER_TASK:-32}"
echo "Memory:             ${SLURM_MEM_PER_NODE:-unknown}M"
echo "Container:          ${SIF_PATH}"
echo "============================================"
echo ""

# Bind mounts: source data, destination data, and host source code.
BIND_ARG="${SRC_ROOT}:${SRC_ROOT},${DST_ROOT}:${DST_ROOT},${REPO_DIR}/src:/workspace/src,${REPO_DIR}/scripts:/workspace/scripts"

export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/tmp}"

# Use dask threaded scheduler (no distributed cluster) to avoid worker OOM.
# read_time_chunks=50 keeps memory bounded for large datasets.
"${APPTAINER_BIN}" exec \
  --bind "${BIND_ARG}" \
  --pwd /workspace \
  "${SIF_PATH}" \
  bash -lc "
    set -euo pipefail
    export PYTHONPATH=/workspace/src:\${PYTHONPATH:-}
    export PYTHONUNBUFFERED=1
    export DASK_NUM_WORKERS=${SLURM_CPUS_PER_TASK:-32}

    /workspace/.venv/bin/python /workspace/scripts/clone_data.py \
      '${DST_ROOT}' \
      --src_root '${SRC_ROOT}' \
      --compact_variables \
      --read_time_chunks 10 \
      ${SOURCES_ARGS}
  "

echo ""
echo "Done. Compacted data written to ${DST_ROOT}"
