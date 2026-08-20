#!/bin/bash
#SBATCH -p mit_quicktest
#SBATCH --job-name=bench_multicache_scaling
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96GB
#SBATCH --time=15:00
#SBATCH -o /orcd/home/002/codycruz/Ocean_Emulator/logs/%x-%j.out
#SBATCH -e /orcd/home/002/codycruz/Ocean_Emulator/logs/%x-%j.out

# Measure how the chunk-first builder scales with threads, on a node where the
# threads actually exist. The login node caps at 2 cores, so every timing taken
# there is a 2-core timing whatever --workers says.
#
# Writes 8 timesteps per setting into a throwaway store; ~15 min total.

set -euo pipefail
module load miniforge/24.3.0-0
cd /orcd/home/002/codycruz/Ocean_Emulator

export PYTHONPATH="/orcd/home/002/codycruz/Ocean_Emulator/src:/orcd/home/002/codycruz/Ocean_Emulator/.venv/lib/python3.11/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="/orcd/home/002/codycruz/Ocean_Emulator/.venv/bin/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

OUT="${OUT:-/orcd/data/abodner/002/cody/LLC_patch/1104-4-tile-test/SCALETEST}"
TILES='[[1,1068,2172,1068,2172],[1,2148,3252,1068,2172],[1,1068,2172,2148,3252],[1,2148,3252,2148,3252]]'
STEPS="${STEPS:-8}"
WORKER_LIST="${WORKER_LIST:-4 8 16 32 64}"

COMMON=(
  --source /orcd/data/abodner/003/LLC4320/LLC4320
  --means /orcd/data/abodner/002/cody/LLC_means_stds/var_96_LLC_means.zarr
  --stds  /orcd/data/abodner/002/cody/LLC_means_stds/var_96_LLC_stds.zarr
  --output-root "${OUT}" --tiles "${TILES}"
  --train-start 2011-09-13 --train-end 2011-09-13
  --val-start 2011-09-14 --val-end 2011-09-14
)

echo "node=$(hostname) cores_available=$(${PYTHON_BIN} -c 'import os;print(len(os.sched_getaffinity(0)))')"
echo "mem=$(free -g | awk 'NR==2{print $2}') GB   steps per setting=${STEPS}"

"${PYTHON_BIN}" scripts/build_multiple_llc_patch_cache_compressed.py "${COMMON[@]}" \
  --init --overwrite 2>&1 | grep -E "INFO" | tail -2

for w in ${WORKER_LIST}; do
  echo "======== workers=${w} ========"
  /usr/bin/time -f "  peak RSS %M KB" \
    "${PYTHON_BIN}" scripts/build_multiple_llc_patch_cache_compressed.py "${COMMON[@]}" \
      --fill --time-index-start 0 --time-index-stop "${STEPS}" \
      --workers "${w}" --log-every "${STEPS}" 2>&1 | grep -E "s/step|read plan|peak RSS"
done
