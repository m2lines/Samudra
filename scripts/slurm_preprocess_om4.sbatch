#!/bin/bash

# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

# Slurm CPU harness for the OM4 preprocessing pipeline on NYU Torch HPC.
#
# This runs `python -m ocean_preprocessing om4 ...` on a single big CPU node using
# a *local* Dask cluster (LocalCluster) instead of coiled. Running on Torch keeps
# all reads/writes on the NYU<->OSN network, avoiding the cloud egress cost that a
# coiled (cloud-region) cluster would incur streaming raw OM4 in and pushing the
# processed Zarr back out.
#
# Environment provisioning: a miniforge/mamba env named `ocean_preprocessing`
# (created once on /scratch from data/mamba_env.yaml) is activated in-job. The
# raw data is streamed directly from the OSN pod via s3fs (no local staging).
#
# --- Required env vars -------------------------------------------------------
#   RESOLUTION   one of: twodeg | twodeg_filter | onedeg | onedeg_filter | halfdeg | quarterdeg
#   OUTPUT_PATH  full s3 zarr path for the result, OR set OUTPUT_BASE (below).
#
# --- Common env vars ---------------------------------------------------------
#   OUTPUT_BASE  s3 prefix; OUTPUT_PATH defaults to
#                  ${OUTPUT_BASE}/om4_${RESOLUTION}/OM4.zarr
#                e.g. export OUTPUT_BASE="s3://m2lines-pubs/Samudra/v$(date +%Y-%m)"
#   RAW_ROOT     source prefix (default: s3://m2lines-pubs/Samudra/raw)
#   PREPROC_ENV  conda env name (default: ocean_preprocessing)
#   CONDA_SH     path to conda.sh (autodetected from $CONDA_EXE / common dirs)
#   REPO_DIR     repo checkout (default: $SLURM_SUBMIT_DIR); we cd here so the
#                pipeline can stamp git-hash provenance and find data/.
#   N_WORKERS    LocalCluster workers (default: $SLURM_CPUS_PER_TASK, capped 32)
#   THREADS_PER_WORKER  threads per worker (default: 1)
#   MEMORY_LIMIT per-worker memory limit passed to LocalCluster (e.g. "40GB");
#                unset lets Dask split the cgroup memory across workers.
#   SKIP_VALIDATION  1 (default) skips deep QC + rotated-vector plots (headless);
#                    set 0 to validate.
#   EXTRA_ARGS   extra flags appended verbatim to the CLI (e.g. --small_run).
#
# Credentials (required to WRITE to OSN; export before sbatch or in your env):
#   export FSSPEC_S3_ENDPOINT_URL=https://nyu1.osn.mghpcc.org/
#   export AWS_ACCESS_KEY_ID=...
#   export AWS_SECRET_ACCESS_KEY=...
#
# --- Example submission ------------------------------------------------------
#   export FSSPEC_S3_ENDPOINT_URL=https://nyu1.osn.mghpcc.org/
#   export AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=...
#   export RESOLUTION=onedeg
#   export OUTPUT_BASE="s3://m2lines-pubs/Samudra/v$(date +%Y-%m)"
#   sbatch scripts/slurm_preprocess_om4.sbatch
#
# The #SBATCH defaults below target Torch's `cs` CPU partition (128-core,
# 513 GB, 184 nodes -- starts promptly). The pipeline is per-timestep chunked,
# so even the 0.25 deg run fits in 513 GB with a bounded worker count; `cs`
# suffices for all resolutions. Per-resolution overrides (CLI flags win over the
# directives below):
#   twodeg / twodeg_filter : --cpus-per-task=32  --mem=240G --time=08:00:00  (coarsest, fastest)
#   onedeg / onedeg_filter : --cpus-per-task=64  --mem=480G --time=12:00:00
#   halfdeg                : --cpus-per-task=128 --mem=490G --time=1-00:00:00
#   quarterdeg             : --cpus-per-task=128 --mem=490G --time=2-00:00:00 \
#                            (also set N_WORKERS=16 to cap peak memory; if it
#                             still OOMs, fall back to --partition=cl for a
#                             3 TB node -- but that partition queues for days)
# The big-memory `cl` partition (7 nodes, 3 TB) is a fallback only; it is
# heavily contended and can queue for days.

#SBATCH --job-name=om4-preprocess
#SBATCH --output=slurm-%j.out
#SBATCH --open-mode=append
#SBATCH --account=torch_pr_347_lzanna
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=1-00:00:00

set -euo pipefail

# --- Resolve inputs ----------------------------------------------------------
RESOLUTION="${RESOLUTION:-}"
if [[ -z "${RESOLUTION}" ]]; then
  echo "ERROR: RESOLUTION is required (onedeg | onedeg_filter | halfdeg | quarterdeg)." >&2
  exit 2
fi

# Resolution -> target grid + whether spatial filtering is applied.
# The target-grid basename also drives the pipeline's automatic filter scale.
case "${RESOLUTION}" in
  twodeg)        GRID="gaussian_grid_90_by_180";   FILTER_FLAG="--skip_spatial_filtering" ;;
  twodeg_filter) GRID="gaussian_grid_90_by_180";   FILTER_FLAG="" ;;   # filtering ON (scale 36)
  onedeg)        GRID="gaussian_grid_180_by_360";  FILTER_FLAG="--skip_spatial_filtering" ;;
  onedeg_filter) GRID="gaussian_grid_180_by_360";  FILTER_FLAG="" ;;   # filtering ON (scale 18)
  halfdeg)       GRID="gaussian_grid_360_by_720";  FILTER_FLAG="--skip_spatial_filtering" ;;
  quarterdeg)    GRID="gaussian_grid_720_by_1440"; FILTER_FLAG="--skip_spatial_filtering" ;;
  *)
    echo "ERROR: unknown RESOLUTION='${RESOLUTION}'." >&2
    echo "Expected one of: twodeg | twodeg_filter | onedeg | onedeg_filter | halfdeg | quarterdeg." >&2
    exit 2 ;;
esac

RAW_ROOT="${RAW_ROOT:-s3://m2lines-pubs/Samudra/raw}"
DATA_PATH="${RAW_ROOT}/om4_5daily.zarr"
NATIVE_GRID_PATH="${RAW_ROOT}/ocean_static_no_mask_table.zarr"
MOSAIC_PATH="${RAW_ROOT}/grids/ocean_hgrid.zarr"
TARGET_GRID_PATH="${RAW_ROOT}/grids/${GRID}.zarr"

OUTPUT_PATH="${OUTPUT_PATH:-}"
if [[ -z "${OUTPUT_PATH}" ]]; then
  if [[ -z "${OUTPUT_BASE:-}" ]]; then
    echo "ERROR: set OUTPUT_PATH, or OUTPUT_BASE to derive it." >&2
    echo "Example: export OUTPUT_BASE=\"s3://m2lines-pubs/Samudra/v\$(date +%Y-%m)\"" >&2
    exit 2
  fi
  OUTPUT_PATH="${OUTPUT_BASE%/}/om4_${RESOLUTION}/OM4.zarr"
fi

# --- Credentials preflight ---------------------------------------------------
# This harness targets the NYU OSN pod; default its endpoint (override to retarget).
export FSSPEC_S3_ENDPOINT_URL="${FSSPEC_S3_ENDPOINT_URL:-https://nyu1.osn.mghpcc.org/}"
: "${AWS_ACCESS_KEY_ID:?OSN write key required (AWS_ACCESS_KEY_ID)}"
: "${AWS_SECRET_ACCESS_KEY:?OSN write secret required (AWS_SECRET_ACCESS_KEY)}"
# OSN's S3 does not support AWS request checksums; disable to avoid write errors.
export AWS_REQUEST_CHECKSUM_CALCULATION="${AWS_REQUEST_CHECKSUM_CALCULATION:-when_required}"
export AWS_RESPONSE_CHECKSUM_VALIDATION="${AWS_RESPONSE_CHECKSUM_VALIDATION:-when_required}"

# --- Environment activation --------------------------------------------------
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
  echo "ERROR: could not locate conda.sh. Set CONDA_SH to <miniforge>/etc/profile.d/conda.sh." >&2
  exit 3
fi
# shellcheck source=/dev/null
source "${CONDA_SH}"
conda activate "${PREPROC_ENV}"

# --- Thread-safety + headless plotting (mirror ocean_preprocessing.__main__) --
export BLOSC_NTHREADS=1
export BLOSC_NOLOCK=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

# --- LocalCluster sizing -----------------------------------------------------
DEFAULT_WORKERS="${SLURM_CPUS_PER_TASK:-8}"
if (( DEFAULT_WORKERS > 32 )); then DEFAULT_WORKERS=32; fi
N_WORKERS="${N_WORKERS:-${DEFAULT_WORKERS}}"
THREADS_PER_WORKER="${THREADS_PER_WORKER:-1}"

cluster_flags=(
  --cluster=local
  --n_workers="${N_WORKERS}"
  --threads_per_worker="${THREADS_PER_WORKER}"
)
if [[ -n "${MEMORY_LIMIT:-}" ]]; then
  cluster_flags+=(--memory_limit="${MEMORY_LIMIT}")
fi

pipeline_flags=()
if [[ -n "${FILTER_FLAG}" ]]; then
  pipeline_flags+=("${FILTER_FLAG}")
fi
if [[ "${SKIP_VALIDATION:-1}" == "1" ]]; then
  pipeline_flags+=(--skip_validation)
fi

# --- Run ---------------------------------------------------------------------
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_DIR}"  # so get_git_url_hash() resolves and data/ is importable

echo "Resolution:   ${RESOLUTION}"
echo "Target grid:  ${GRID}  (filter: ${FILTER_FLAG:-on})"
echo "Raw root:     ${RAW_ROOT}"
echo "Output path:  ${OUTPUT_PATH}"
echo "Conda env:    ${PREPROC_ENV}  (via ${CONDA_SH})"
echo "LocalCluster: n_workers=${N_WORKERS} threads_per_worker=${THREADS_PER_WORKER} memory_limit=${MEMORY_LIMIT:-auto}"
echo "Repo dir:     ${REPO_DIR}"

set -x
python -m ocean_preprocessing om4 \
  "${DATA_PATH}" \
  "${NATIVE_GRID_PATH}" \
  "${MOSAIC_PATH}" \
  "${TARGET_GRID_PATH}" \
  --output_path="${OUTPUT_PATH}" \
  "${cluster_flags[@]}" \
  "${pipeline_flags[@]}" \
  ${EXTRA_ARGS:-}
set +x

echo "OM4 preprocessing complete: ${OUTPUT_PATH}"
echo "Next: normalization datasets ->"
echo "  RESOLUTION=${RESOLUTION} OUTPUT_PATH=${OUTPUT_PATH} sbatch scripts/slurm_make_norm_om4.sbatch"
