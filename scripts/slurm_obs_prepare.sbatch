#!/bin/bash

# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

# Slurm CPU harness for the observation-preparation pipeline on NYU Torch HPC.
#
# This runs `python -m ocean_preprocessing.obs_preprocessing prepare all ...`,
# turning a raw observation archive on /scratch into the analysis-ready stores
# that `samudra.metrics` scores rollouts against.
#
# Fill the raw archive first with `scripts/slurm_obs_download.sbatch`.
#
# --- Required env vars -------------------------------------------------------
#   RAW_ROOT     directory holding the raw archive, with duacs/, oisst/ and
#                argo-iap/ subdirectories (as written by the download stage)
#   OUTPUT_ROOT  where to write duacs.zarr, oisst.zarr, oisst_daily.zarr and
#                argo-iap.zarr. Local path or s3:// prefix.
#
# --- Common env vars ---------------------------------------------------------
#   OM4_PATH     store whose 5-day calendar the observations align onto
#                (default: the public one-degree OM4; read anonymously)
#   START_DATE   first target timestamp (default 2014-10-20)
#   END_DATE     last target timestamp  (default 2022-12-24)
#   PRODUCT      all | oisst | argo_iap | duacs  (default: all)
#   PREPROC_ENV  conda env name (default: ocean_preprocessing)
#   CONDA_SH     path to conda.sh (autodetected from $CONDA_EXE / common dirs)
#   REPO_DIR     repo checkout (default: $SLURM_SUBMIT_DIR); we cd here so the
#                pipeline can stamp git-hash provenance and find data/.
#   EXTRA_ARGS   extra flags appended verbatim (e.g. --dry_run)
#
# Credentials are only needed to WRITE results to OSN:
#   export FSSPEC_S3_ENDPOINT_URL=https://nyu1.osn.mghpcc.org/
#   export AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=...
#
# --- Example submission ------------------------------------------------------
#   export RAW_ROOT=/scratch/$USER/data/obs_raw
#   export OUTPUT_ROOT=/scratch/$USER/data/obs
#   sbatch scripts/slurm_obs_prepare.sbatch
#
# Sizing notes: DUACS dominates. Its raw daily record over 2014-2022 at 0.125
# degrees is a few hundred GB, and the centered rolling mean streams all of it,
# so give this job real memory and time. OISST and ARGO are comparatively small;
# set PRODUCT to run them alone if DUACS is not ready yet.

#SBATCH --job-name=obs-prepare
#SBATCH --output=slurm-%j.out
#SBATCH --open-mode=append
#SBATCH --account=torch_pr_347_lzanna
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=1-00:00:00

set -euo pipefail

RAW_ROOT="${RAW_ROOT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
if [[ -z "${RAW_ROOT}" || -z "${OUTPUT_ROOT}" ]]; then
  echo "ERROR: RAW_ROOT and OUTPUT_ROOT are both required." >&2
  echo "  export RAW_ROOT=/scratch/\$USER/data/obs_raw" >&2
  echo "  export OUTPUT_ROOT=/scratch/\$USER/data/obs" >&2
  exit 2
fi

PRODUCT="${PRODUCT:-all}"
case "${PRODUCT}" in
  all|oisst|argo_iap|duacs) ;;
  *) echo "ERROR: PRODUCT must be one of: all | oisst | argo_iap | duacs" >&2; exit 2 ;;
esac

OM4_PATH="${OM4_PATH:-s3://m2lines-pubs/Samudra/v2026-07/om4_onedeg/OM4.zarr}"
START_DATE="${START_DATE:-2014-10-20}"
END_DATE="${END_DATE:-2022-12-24}"

# --- Credentials -------------------------------------------------------------
# The OM4 time axis is public, so reads need no keys; writes to OSN do.
export FSSPEC_S3_ENDPOINT_URL="${FSSPEC_S3_ENDPOINT_URL:-https://nyu1.osn.mghpcc.org/}"
if [[ "${OUTPUT_ROOT}" == s3://* ]]; then
  : "${AWS_ACCESS_KEY_ID:?OSN write key required to write to ${OUTPUT_ROOT}}"
  : "${AWS_SECRET_ACCESS_KEY:?OSN write secret required to write to ${OUTPUT_ROOT}}"
fi
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

export BLOSC_NTHREADS=1
export BLOSC_NOLOCK=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_DIR}/data"  # so get_git_url_hash() resolves and the package imports

echo "Product:      ${PRODUCT}"
echo "Raw root:     ${RAW_ROOT}"
echo "Output root:  ${OUTPUT_ROOT}"
echo "OM4 calendar: ${OM4_PATH}"
echo "Window:       ${START_DATE} .. ${END_DATE}"
echo "Conda env:    ${PREPROC_ENV}  (via ${CONDA_SH})"

date_flags=()
if [[ "${PRODUCT}" != "argo_iap" ]]; then
  date_flags+=(--om4_path="${OM4_PATH}" --start_date="${START_DATE}" --end_date="${END_DATE}")
fi

set -x
python -m ocean_preprocessing.obs_preprocessing prepare "${PRODUCT}" \
  --raw_root="${RAW_ROOT}" \
  --output_root="${OUTPUT_ROOT}" \
  ${date_flags[@]+"${date_flags[@]}"} \
  ${EXTRA_ARGS:-}
