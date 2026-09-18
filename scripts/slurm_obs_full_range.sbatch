#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
# See docs/observations-full-range.md. This harness never uploads to OSN.
#SBATCH --job-name=obs-full-range
#SBATCH --account=torch_pr_347_lzanna
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --open-mode=append

set -euo pipefail
: "${STAGE:?Set STAGE=setup|discover|download|prepare|validate}"
: "${REPO_DIR:?Set REPO_DIR to a fixed checkout containing these edits}"
PRODUCT="${PRODUCT:-duacs}"
case "$PRODUCT" in duacs|oisst|argo-iap) ;; *) echo "Invalid PRODUCT: $PRODUCT" >&2; exit 2 ;; esac
WORK_ROOT="${WORK_ROOT:-/scratch/${USER}/data/obs_full_range}"
RAW_ROOT="${RAW_ROOT:-${WORK_ROOT}/raw}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/prepared}"
MANIFEST_PATH="${MANIFEST_PATH:-${WORK_ROOT}/manifests/${PRODUCT}.json}"
OBS_ENV="${OBS_ENV:-${WORK_ROOT}/venv}"
OBS_PYTHON="${OBS_PYTHON:-${OBS_ENV}/bin/python}"
mkdir -p "$WORK_ROOT"
# Protect a staging store or raw archive from concurrent retries.
exec 9>"${WORK_ROOT}/.${STAGE}-${PRODUCT}.lock"
flock -n 9 || { echo "Another $STAGE job for $PRODUCT is active" >&2; exit 1; }
export PYTHONUNBUFFERED=1 BLOSC_NTHREADS=1 BLOSC_NOLOCK=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH="${REPO_DIR}/data${PYTHONPATH:+:${PYTHONPATH}}"
export PIP_CACHE_DIR="${WORK_ROOT}/pip-cache"
cd "${REPO_DIR}/data"
if [[ "$STAGE" == setup ]]; then
    python3 -m venv "$OBS_ENV"
    "$OBS_PYTHON" -m pip install -r "${REPO_DIR}/scripts/requirements-observations.txt"
    "$OBS_PYTHON" -m pip freeze > "${WORK_ROOT}/environment-freeze.txt"
    "$OBS_PYTHON" -c 'from ocean_preprocessing.obs_preprocessing.full_range import FullRange; print("Observation environment ready")'
    exit 0
fi
[[ -x "$OBS_PYTHON" ]] || { echo "Run STAGE=setup first" >&2; exit 2; }
args=()
case "$STAGE" in
    discover)
        args=(discover --product="$PRODUCT" --manifest_path="$MANIFEST_PATH")
        if [[ "$PRODUCT" == duacs && "${INCLUDE_SSH:-0}" == 1 ]]; then args+=(--include_ssh=True); fi
        ;;
    download)
        args=(download --manifest_path="$MANIFEST_PATH" --raw_root="$RAW_ROOT" --workers="${WORKERS:-4}")
        if [[ -n "${COPERNICUS_CREDENTIALS_FILE:-}" ]]; then
            args+=(--credentials_file="$COPERNICUS_CREDENTIALS_FILE")
        fi
        ;;
    prepare)
        args=(prepare --manifest_path="$MANIFEST_PATH" --raw_root="$RAW_ROOT" --output_root="$OUTPUT_ROOT" --threads="${SLURM_CPUS_PER_TASK:-4}")
        ;;
    validate)
        args=(validate --manifest_path="$MANIFEST_PATH" --store="${OUTPUT_ROOT}/${PRODUCT}.zarr")
        ;;
    *) echo "Invalid STAGE: $STAGE" >&2; exit 2 ;;
esac
printf 'Stage=%s Product=%s Manifest=%s\n' "$STAGE" "$PRODUCT" "$MANIFEST_PATH"
# Never enable set -x here: credentials can be inherited in the environment.
"$OBS_PYTHON" -m ocean_preprocessing.obs_preprocessing full_range "${args[@]}"
