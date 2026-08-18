#!/bin/bash

# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

# Slurm CPU harness for building normalization datasets (OM4_means.zarr /
# OM4_stds.zarr) alongside a processed OM4.zarr on the OSN pod.
#
# `ocean_preprocessing/make_norm_datasets.py` already uses a local Dask cluster,
# so this wrapper just supplies the OSN endpoint + credentials and the s3 path.
# Run it after scripts/slurm_preprocess_om4.sbatch finishes for a resolution.
#
# --- Required env vars -------------------------------------------------------
#   OUTPUT_PATH  s3 path to the processed OM4.zarr, OR set OUTPUT_BASE+RESOLUTION.
#
# --- Common env vars ---------------------------------------------------------
#   OUTPUT_BASE + RESOLUTION  derive OUTPUT_PATH as
#                               ${OUTPUT_BASE}/om4_${RESOLUTION}/OM4.zarr
#   PREPROC_ENV  conda env name (default: ocean_preprocessing)
#   CONDA_SH     path to conda.sh (autodetected from $CONDA_EXE / common dirs)
#   REPO_DIR     repo checkout (default: $SLURM_SUBMIT_DIR)
#
# Credentials (required to WRITE to OSN):
#   export FSSPEC_S3_ENDPOINT_URL=https://nyu1.osn.mghpcc.org/
#   export AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=...
#
# --- Example -----------------------------------------------------------------
#   export OUTPUT_PATH="s3://m2lines-pubs/Samudra/v2026-07/om4_onedeg/OM4.zarr"
#   sbatch scripts/slurm_make_norm_om4.sbatch
#
# Defaults target Torch's `cs` CPU partition. mean/std reduce over the whole
# dataset; bump --mem/--time for halfdeg/quarterdeg.

#SBATCH --job-name=om4-norm
#SBATCH --output=slurm-%j.out
#SBATCH --open-mode=append
#SBATCH --account=torch_pr_347_lzanna
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=12:00:00

set -euo pipefail

OUTPUT_PATH="${OUTPUT_PATH:-}"
if [[ -z "${OUTPUT_PATH}" ]]; then
  if [[ -z "${OUTPUT_BASE:-}" || -z "${RESOLUTION:-}" ]]; then
    echo "ERROR: set OUTPUT_PATH, or both OUTPUT_BASE and RESOLUTION." >&2
    exit 2
  fi
  OUTPUT_PATH="${OUTPUT_BASE%/}/om4_${RESOLUTION}/OM4.zarr"
fi
if [[ "${OUTPUT_PATH}" != s3://* ]]; then
  echo "ERROR: make_norm_datasets.py requires an s3:// path; got '${OUTPUT_PATH}'." >&2
  exit 2
fi

# This harness targets the NYU OSN pod; default its endpoint (override to retarget).
export FSSPEC_S3_ENDPOINT_URL="${FSSPEC_S3_ENDPOINT_URL:-https://nyu1.osn.mghpcc.org/}"
: "${AWS_ACCESS_KEY_ID:?OSN write key required (AWS_ACCESS_KEY_ID)}"
: "${AWS_SECRET_ACCESS_KEY:?OSN write secret required (AWS_SECRET_ACCESS_KEY)}"
export AWS_REQUEST_CHECKSUM_CALCULATION="${AWS_REQUEST_CHECKSUM_CALCULATION:-when_required}"
export AWS_RESPONSE_CHECKSUM_VALIDATION="${AWS_RESPONSE_CHECKSUM_VALIDATION:-when_required}"

PREPROC_ENV="${PREPROC_ENV:-ocean_preprocessing}"
CONDA_SH="${CONDA_SH:-}"
if [[ -z "${CONDA_SH}" ]]; then
  if [[ -n "${CONDA_EXE:-}" ]]; then
    CONDA_SH="$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh"
  else
    for cand in \
      "${HOME}/miniforge3/etc/profile.d/conda.sh" \
      "/scratch/${USER}/miniforge3/etc/profile.d/conda.sh" \
      "${HOME}/miniconda3/etc/profile.d/conda.sh"; do
      [[ -f "${cand}" ]] && CONDA_SH="${cand}" && break
    done
  fi
fi
if [[ -z "${CONDA_SH}" || ! -f "${CONDA_SH}" ]]; then
  echo "ERROR: could not locate conda.sh. Set CONDA_SH." >&2
  exit 3
fi
# shellcheck source=/dev/null
source "${CONDA_SH}"
conda activate "${PREPROC_ENV}"

export BLOSC_NTHREADS=1
export BLOSC_NOLOCK=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_DIR}/data"

echo "Building means/stds for: ${OUTPUT_PATH}"
set -x
python ocean_preprocessing/make_norm_datasets.py "${OUTPUT_PATH}"
set +x

echo "Normalization datasets written next to ${OUTPUT_PATH}"
