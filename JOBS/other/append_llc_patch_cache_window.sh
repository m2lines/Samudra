#!/bin/bash
#SBATCH -p pi_abodner
#SBATCH -w node2905
#SBATCH --job-name=2026-08-09-append-test-window-llc_patch_cache-4tile
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --time=12:00:00
#SBATCH -o /orcd/home/002/codycruz/Ocean_Emulator/logs/%x-%j.out
#SBATCH -e /orcd/home/002/codycruz/Ocean_Emulator/logs/%x-%j.out

# Extend every tile cache in a group with a later time window, in place.
#
# The caches were built with train+val windows only, so inference past val_end
# has no data to read. This appends a test window rather than rebuilding: a
# rebuild would re-read and re-write the whole 13-month store per tile to add a
# few percent more time.
#
# Append is concatenation, not a merge, so the new window must start at or after
# the stored end. Timestamps already present are dropped, which makes this
# script safe to re-run after a failure.

set -euo pipefail

module load miniforge/24.3.0-0
cd /orcd/home/002/codycruz/Ocean_Emulator

PROJECT_SITE_PACKAGES="/orcd/home/002/codycruz/Ocean_Emulator/.venv/lib/python3.11/site-packages"
export PYTHONPATH="/orcd/home/002/codycruz/Ocean_Emulator/src:${PROJECT_SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="/orcd/home/002/codycruz/Ocean_Emulator/.venv/bin/python"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

SOURCE_ZARR="${SOURCE_ZARR:-/orcd/data/abodner/003/LLC4320/LLC4320}"
MEANS_ZARR="${MEANS_ZARR:-/orcd/data/abodner/002/cody/LLC_means_stds/var_96_LLC_means.zarr}"
STDS_ZARR="${STDS_ZARR:-/orcd/data/abodner/002/cody/LLC_means_stds/var_96_LLC_stds.zarr}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/orcd/data/abodner/002/cody/LLC_patch/720-div-4-test}"

# ============== WINDOW TO APPEND ==============
# LLC4320 runs to 2012-11-15, so this window exists. The caches already hold all
# of 2012-10-14, so those 24 hours are deduped and 408 timestamps are appended.
APPEND_START="${APPEND_START:-2012-10-14}"
APPEND_END="${APPEND_END:-2012-10-31}"
APPEND_LABEL="${APPEND_LABEL:-test}"

# ============== EXISTING STORE IDENTITY ==============
# These reproduce the *original* store filename and must not be changed to the
# append window -- the name encodes the window the store was first built with.
LLC_FACE="${LLC_FACE:-1}"
TRAIN_START="${TRAIN_START:-2011-09-13}"
TRAIN_END="${TRAIN_END:-2012-09-13}"
VAL_START="${VAL_START:-2012-09-14}"
VAL_END="${VAL_END:-2012-10-14}"
FLOAT_TYPE="${FLOAT_TYPE:-float16}"
TIME_CHUNK="${TIME_CHUNK:-1}"
DRY_RUN="${DRY_RUN:-false}"

# ============== TILES ==============
# "i_start i_end j_start j_end" per tile, matching 720-div-4-test.
TILES=(
  "2880 3248 720 1088"
  "2880 3248 1072 1440"
  "3232 3600 720 1088"
  "3232 3600 1072 1440"
)

TRAIN_START_TAG="${TRAIN_START//-/}"
VAL_END_TAG="${VAL_END//-/}"

echo "======== append a time window to every tile cache ========"
echo "output_root:   ${OUTPUT_ROOT}"
echo "append window: ${APPEND_START} -> ${APPEND_END}  (label=${APPEND_LABEL})"
echo "tiles:         ${#TILES[@]}"
echo "dry_run:       ${DRY_RUN}"
echo

for tile in "${TILES[@]}"; do
  read -r I_START I_END J_START J_END <<<"${tile}"
  OUTPUT_NAME="LLC4320_face${LLC_FACE}_i${I_START}-${I_END}_j${J_START}-${J_END}_trainval_ready_${TRAIN_START_TAG}_${VAL_END_TAG}_t${TIME_CHUNK}.zarr"

  echo "-------- ${OUTPUT_NAME} --------"
  if [[ ! -d "${OUTPUT_ROOT}/${OUTPUT_NAME}" ]]; then
    echo "ERROR: no existing store at ${OUTPUT_ROOT}/${OUTPUT_NAME}" >&2
    exit 1
  fi

  ARGS=(
    --source "${SOURCE_ZARR}"
    --means "${MEANS_ZARR}"
    --stds "${STDS_ZARR}"
    --output-root "${OUTPUT_ROOT}"
    --output-name "${OUTPUT_NAME}"
    --face "${LLC_FACE}"
    --i-start "${I_START}"
    --i-end "${I_END}"
    --j-start "${J_START}"
    --j-end "${J_END}"
    --train-start "${TRAIN_START}"
    --train-end "${TRAIN_END}"
    --val-start "${VAL_START}"
    --val-end "${VAL_END}"
    --float-type "${FLOAT_TYPE}"
    --time-chunk "${TIME_CHUNK}"
    --append
    --append-start "${APPEND_START}"
    --append-end "${APPEND_END}"
    --append-label "${APPEND_LABEL}"
  )
  if [[ "${DRY_RUN}" == "true" ]]; then
    ARGS+=(--dry-run)
  fi

  "${PYTHON_BIN}" scripts/build_llc_patch_cache_compressed_train_val.py "${ARGS[@]}"
  echo
done

echo "Done. Every tile cache now covers through ${APPEND_END}."
echo "Inference can now use --inference_time.end up to ${APPEND_END}."
