#!/bin/bash

# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

# Slurm harness to fill the raw observation archive on NYU Torch HPC.
#
# Fetches the products that `samudra.metrics` scores rollouts against:
#   oisst     NOAA OISST v2.1 daily SST          (no credentials)
#   argo_iap  IAP monthly temperature/salinity    (no credentials)
#   duacs     Copernicus Marine daily sea level   (free account required)
#
# Torch compute nodes have general internet egress (verified on the `cs`
# partition), so these run as ordinary batch jobs. The work is network-bound
# rather than CPU-bound, so this asks for few cores and modest memory; raise
# WORKERS rather than --cpus-per-task to go faster.
#
# Every downloader is restartable: files already present at a plausible size are
# skipped, and partial downloads land on `.part` and are only renamed on
# success. Re-submitting after a timeout or a network blip fetches only what is
# still missing, so a long fetch can safely be split across several jobs.
#
# --- Required env vars -------------------------------------------------------
#   RAW_ROOT   directory to fill, e.g. /scratch/$USER/data/obs_raw
#
# --- Common env vars ---------------------------------------------------------
#   PRODUCT      oisst | argo_iap | duacs   (default: oisst)
#   START_YEAR   first year  (oisst default 1982, argo_iap default 1980)
#   END_YEAR     last year   (default 2022)
#   START_DATE   duacs only, default 2014-10-18
#   END_DATE     duacs only, default 2022-12-26
#   WORKERS      concurrent downloads (oisst default 8, argo_iap default 4)
#   PREPROC_ENV  conda env name (default: ocean_preprocessing)
#   CONDA_SH     path to conda.sh (autodetected from $CONDA_EXE / common dirs)
#   REPO_DIR     repo checkout (default: $SLURM_SUBMIT_DIR)
#   EXTRA_ARGS   extra flags appended verbatim (e.g. --dry_run, --overwrite)
#
# DUACS additionally requires a free Copernicus Marine account
# (https://data.marine.copernicus.eu/register). Export before submitting:
#   export COPERNICUSMARINE_SERVICE_USERNAME=...
#   export COPERNICUSMARINE_SERVICE_PASSWORD=...
# Note that `sbatch` propagates your environment by default, so these reach the
# job without appearing in this file or in the job script.
#
# --- Example submission ------------------------------------------------------
#   export RAW_ROOT=/scratch/$USER/data/obs_raw
#   PRODUCT=oisst    sbatch scripts/slurm_obs_download.sbatch
#   PRODUCT=argo_iap sbatch scripts/slurm_obs_download.sbatch
#   PRODUCT=duacs    sbatch scripts/slurm_obs_download.sbatch
#
# Sizing: OISST 1982-2022 is ~15k files / ~25 GB. ARGO-IAP 1980-2022 is ~1k
# files. DUACS 2014-2022 with the four velocity variables is a few hundred GB
# and is by far the longest job; it writes one zarr per calendar year so an
# interrupted run resumes at year granularity.

#SBATCH --job-name=obs-download
#SBATCH --output=slurm-%j.out
#SBATCH --open-mode=append
#SBATCH --account=torch_pr_347_lzanna
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00

set -euo pipefail

RAW_ROOT="${RAW_ROOT:-}"
if [[ -z "${RAW_ROOT}" ]]; then
  echo "ERROR: RAW_ROOT is required, e.g. export RAW_ROOT=/scratch/\$USER/data/obs_raw" >&2
  exit 2
fi

PRODUCT="${PRODUCT:-oisst}"
case "${PRODUCT}" in
  oisst|argo_iap|duacs) ;;
  *) echo "ERROR: PRODUCT must be one of: oisst | argo_iap | duacs" >&2; exit 2 ;;
esac

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

export PYTHONUNBUFFERED=1

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_DIR}/data"

echo "Product:  ${PRODUCT}"
echo "Raw root: ${RAW_ROOT}"
echo "Conda env: ${PREPROC_ENV}  (via ${CONDA_SH})"

case "${PRODUCT}" in
  oisst)
    flags=(
      --output_dir="${RAW_ROOT}/oisst"
      --start_year="${START_YEAR:-1982}"
      --end_year="${END_YEAR:-2022}"
      --workers="${WORKERS:-8}"
    )
    ;;
  argo_iap)
    flags=(
      --output_dir="${RAW_ROOT}/argo-iap"
      --start_year="${START_YEAR:-1980}"
      --end_year="${END_YEAR:-2022}"
      --workers="${WORKERS:-4}"
    )
    ;;
  duacs)
    : "${COPERNICUSMARINE_SERVICE_USERNAME:?Copernicus Marine username required for DUACS}"
    : "${COPERNICUSMARINE_SERVICE_PASSWORD:?Copernicus Marine password required for DUACS}"
    flags=(
      --output_dir="${RAW_ROOT}/duacs"
      --start_date="${START_DATE:-2014-10-18}"
      --end_date="${END_DATE:-2022-12-26}"
    )
    ;;
esac

set -x
python -m ocean_preprocessing.obs_preprocessing download "${PRODUCT}" \
  "${flags[@]}" \
  ${EXTRA_ARGS:-}
