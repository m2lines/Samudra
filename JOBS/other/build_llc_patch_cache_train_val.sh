#!/bin/bash
#SBATCH -p pi_abodner
#SBATCH -w node4006
#SBATCH --job-name=2026-07-01-full_yr-llc_patch_cache-face1-2880-4320-j0-1440_trainval_ready--bfloat16-compressed-quad-Agulhas
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --time=48:00:00
#SBATCH -o /orcd/home/002/codycruz/Ocean_Emulator/logs/%x-%j.out
#SBATCH -e /orcd/home/002/codycruz/Ocean_Emulator/logs/%x-%j.out


set -euo pipefail

module load miniforge/24.3.0-0

cd /orcd/home/002/codycruz/Ocean_Emulator

PROJECT_SITE_PACKAGES="/orcd/home/002/codycruz/Ocean_Emulator/.venv/lib/python3.11/site-packages"
export PYTHONPATH="/orcd/home/002/codycruz/Ocean_Emulator/src:${PROJECT_SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="/orcd/home/002/codycruz/Ocean_Emulator/.venv/bin/python"

# Safer default threading for large packed-cache writes.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

SOURCE_ZARR="${SOURCE_ZARR:-/orcd/data/abodner/003/LLC4320/LLC4320}"
MEANS_ZARR="${MEANS_ZARR:-/orcd/data/abodner/002/cody/LLC_means_stds/var_96_LLC_means.zarr}"
STDS_ZARR="${STDS_ZARR:-/orcd/data/abodner/002/cody/LLC_means_stds/var_96_LLC_stds.zarr}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/orcd/data/abodner/002/cody/LLC_patch}" # save to storage
#OUTPUT_ROOT="${OUTPUT_ROOT:-/orcd/scratch/codycruz/LLC_patch}" # save to scratch

LLC_FACE="${LLC_FACE:-1}"
LLC_I_START="${LLC_I_START:-2880}"
LLC_I_END="${LLC_I_END:-4320}"
LLC_J_START="${LLC_J_START:-0}"
LLC_J_END="${LLC_J_END:-1440}"

TRAIN_START="${TRAIN_START:-2011-09-13}"
TRAIN_END="${TRAIN_END:-2012-09-13}"
VAL_START="${VAL_START:-2012-09-14}"
VAL_END="${VAL_END:-2012-10-14}"

# TRAIN_START="${TRAIN_START:-2011-09-13}"
# TRAIN_END="${TRAIN_END:-2011-09-20}"
# VAL_START="${VAL_START:-2011-09-20}"
# VAL_END="${VAL_END:-2011-09-27}"

FLOAT_TYPE="${FLOAT_TYPE:-float16}"

TIME_CHUNK="${TIME_CHUNK:-1}"
TRAIN_START_TAG="${TRAIN_START//-/}"
VAL_END_TAG="${VAL_END//-/}"
OUTPUT_NAME="${OUTPUT_NAME:-LLC4320_face${LLC_FACE}_i${LLC_I_START}-${LLC_I_END}_j${LLC_J_START}-${LLC_J_END}_trainval_ready_${TRAIN_START_TAG}_${VAL_END_TAG}_t${TIME_CHUNK}.zarr}"
OUTPUT_PATH="${OUTPUT_ROOT}/${OUTPUT_NAME}"

OVERWRITE="${OVERWRITE:-false}"
DRY_RUN="${DRY_RUN:-false}"
QUEUE_TRAIN_AFTER_BUILD="${QUEUE_TRAIN_AFTER_BUILD:-false}"
TRAIN_JOB_SCRIPT="${TRAIN_JOB_SCRIPT:-JOBS/train_llc.sh}"

echo "======== build LLC packed train+val cache ========"
echo "source=${SOURCE_ZARR}"
echo "means=${MEANS_ZARR}"
echo "stds=${STDS_ZARR}"
echo "output_root=${OUTPUT_ROOT}"
echo "output_name=${OUTPUT_NAME}"
echo "output_path=${OUTPUT_PATH}"
echo "face=${LLC_FACE}, i=[${LLC_I_START}:${LLC_I_END}), j=[${LLC_J_START}:${LLC_J_END})"
echo "train=[${TRAIN_START}:${TRAIN_END}], val=[${VAL_START}:${VAL_END}]"
echo "time_chunk=${TIME_CHUNK}"
echo "overwrite=${OVERWRITE}, dry_run=${DRY_RUN}"
echo "queue_train_after_build=${QUEUE_TRAIN_AFTER_BUILD}"
echo "note: this writes packed prognostic/boundary arrays plus embedded stats/masks"

ARGS=(
  --source "${SOURCE_ZARR}"
  --means "${MEANS_ZARR}"
  --stds "${STDS_ZARR}"
  --output-root "${OUTPUT_ROOT}"
  --output-name "${OUTPUT_NAME}"
  --face "${LLC_FACE}"
  --i-start "${LLC_I_START}"
  --i-end "${LLC_I_END}"
  --j-start "${LLC_J_START}"
  --j-end "${LLC_J_END}"
  --train-start "${TRAIN_START}"
  --train-end "${TRAIN_END}"
  --val-start "${VAL_START}"
  --val-end "${VAL_END}"
  --float-type "${FLOAT_TYPE}"
  --time-chunk "${TIME_CHUNK}"
)

if [[ "${OVERWRITE}" == "true" ]]; then
  ARGS+=(--overwrite)
fi
if [[ "${DRY_RUN}" == "true" ]]; then
  ARGS+=(--dry-run)
fi

if [[ "${QUEUE_TRAIN_AFTER_BUILD}" == "true" ]]; then
  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: QUEUE_TRAIN_AFTER_BUILD=true requires running this file under sbatch." >&2
    exit 1
  fi

  train_submit_output="$(
    sbatch \
      --dependency="afterok:${SLURM_JOB_ID}" \
      --export="ALL,DATA_LOCATION_OVERRIDE=${OUTPUT_PATH},LLC_FACE=${LLC_FACE},LLC_I_START=${LLC_I_START},LLC_I_END=${LLC_I_END},LLC_J_START=${LLC_J_START},LLC_J_END=${LLC_J_END}" \
      "${TRAIN_JOB_SCRIPT}"
  )"
  echo "queued dependent train job: ${train_submit_output}"
fi

"${PYTHON_BIN}" scripts/build_llc_patch_cache_compressed_train_val.py "${ARGS[@]}"
