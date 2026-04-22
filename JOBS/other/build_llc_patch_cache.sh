#!/bin/bash
#SBATCH -p pi_abodner
#SBATCH --job-name=2026-03-21-llc_patch_cache-face1-i2880-3600-j720-1440
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --time=24:00:00
#SBATCH -o /orcd/home/002/codycruz/Ocean_Emulator/logs/%x-%j.out
#SBATCH -e /orcd/home/002/codycruz/Ocean_Emulator/logs/%x-%j.out

set -euo pipefail

module load miniforge/24.3.0-0

cd /orcd/home/002/codycruz/Ocean_Emulator
uv sync --dev

# Safer default threading for heavy rechunk writes.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

SOURCE_ZARR="${SOURCE_ZARR:-/orcd/data/abodner/003/LLC4320/LLC4320}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/orcd/data/abodner/002/cody/LLC_patch}"
OUTPUT_NAME="${OUTPUT_NAME:-}"

LLC_FACE="${LLC_FACE:-1}"
LLC_I_START="${LLC_I_START:-2880}"
LLC_I_END="${LLC_I_END:-3600}"
LLC_J_START="${LLC_J_START:-720}"
LLC_J_END="${LLC_J_END:-1440}"

TIME_CHUNK="${TIME_CHUNK:-1}"
I_CHUNK="${I_CHUNK:-720}"
J_CHUNK="${J_CHUNK:-720}"
K_CHUNK="${K_CHUNK:-51}"
KP1_CHUNK="${KP1_CHUNK:-52}"

INCLUDE_ALL_VARS="${INCLUDE_ALL_VARS:-false}"
INCLUDE_VARS="${INCLUDE_VARS:-U,W,V,Theta,Salt,Eta,oceTAUX,oceTAUY,oceQnet,mask_c}"

OVERWRITE="${OVERWRITE:-false}"
RESUME="${RESUME:-true}"
DRY_RUN="${DRY_RUN:-false}"

echo "======== build LLC patch cache ========"
echo "source=${SOURCE_ZARR}"
echo "output_root=${OUTPUT_ROOT}"
echo "face=${LLC_FACE}, i=[${LLC_I_START}:${LLC_I_END}), j=[${LLC_J_START}:${LLC_J_END})"
echo "chunks: time=${TIME_CHUNK}, i=${I_CHUNK}, j=${J_CHUNK}, k=${K_CHUNK}, k_p1=${KP1_CHUNK}"
echo "include_all_vars=${INCLUDE_ALL_VARS}"
echo "include_vars=${INCLUDE_VARS}"
echo "overwrite=${OVERWRITE}, resume=${RESUME}, dry_run=${DRY_RUN}"

ARGS=(
  --source "${SOURCE_ZARR}"
  --output-root "${OUTPUT_ROOT}"
  --face "${LLC_FACE}"
  --i-start "${LLC_I_START}"
  --i-end "${LLC_I_END}"
  --j-start "${LLC_J_START}"
  --j-end "${LLC_J_END}"
  --time-chunk "${TIME_CHUNK}"
  --i-chunk "${I_CHUNK}"
  --j-chunk "${J_CHUNK}"
  --k-chunk "${K_CHUNK}"
  --kp1-chunk "${KP1_CHUNK}"
)

if [[ -n "${OUTPUT_NAME}" ]]; then
  ARGS+=(--output-name "${OUTPUT_NAME}")
fi

if [[ "${INCLUDE_ALL_VARS}" == "true" ]]; then
  ARGS+=(--include-all-vars)
else
  IFS=',' read -r -a INCLUDE_VARS_ARR <<< "${INCLUDE_VARS}"
  ARGS+=(--include-vars "${INCLUDE_VARS_ARR[@]}")
fi

if [[ "${OVERWRITE}" == "true" ]]; then
  ARGS+=(--overwrite)
fi
if [[ "${RESUME}" == "false" ]]; then
  ARGS+=(--no-resume)
fi
if [[ "${DRY_RUN}" == "true" ]]; then
  ARGS+=(--dry-run)
fi

uv run scripts/build_llc_patch_cache.py "${ARGS[@]}"
