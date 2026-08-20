#!/bin/bash
#SBATCH -p mit_normal
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --job-name=2026-08-20-llc_multicache
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH -o /orcd/home/002/codycruz/Ocean_Emulator/logs/%x-%A_%a-%j.out
#SBATCH -e /orcd/home/002/codycruz/Ocean_Emulator/logs/%x-%A_%a-%j.out

# ============================================================================
# Build SEVERAL LLC patch caches in one chunk-first pass.
#
# The per-cache builder is cache-first: one job per output cache, each opening
# whatever source chunks its own tile needs. Over a 2x2 tile block that reads
# the same 720x720 chunks 36 times per 3D variable per timestep where 16 would
# do, and reads every full-globe 2D chunk four times where one would do.
#
# This one inverts the loop: work out which source chunks the whole tile SET
# needs, open each once, scatter the pieces into every cache that wants them.
# Each source byte is read exactly once -- the floor for any tool. The saving
# grows with tile count, so it matters far more at face/globe scale than here.
#
# Parallelism is by TIME, not by cache. Splitting by cache would reintroduce the
# duplicate reads this exists to remove; splitting by time adds no redundancy at
# all, because a zarr chunk is keyed by its time index, so two jobs writing
# different time ranges touch different files and never race.
#
# TWO-STEP SUBMISSION. Init creates the stores (fast, minutes); the array fills
# them. The dependency matters: a fill job cannot open a store that is not there.
#
#     init=$(MODE=init sbatch --parsable JOBS/other/build_multiple_llc_patch_caches.sh)
#     MODE=fill sbatch --dependency=afterok:${init} JOBS/other/build_multiple_llc_patch_caches.sh
#
# Or, for a short test window, one job start to finish:
#     MODE=all TIME_SPLITS=1 sbatch JOBS/other/build_multiple_llc_patch_caches.sh
#
# MEMORY. Peak is roughly:
#     (one source chunk per read thread)  WORKERS x 105 MiB
#   + (one output buffer per tile)        NUM_TILES x channels x H x W x 2 B
#   + (one inflated 2D globe plane)       ~1 GiB
# which is ~8 GiB for four 1104 tiles at 16 threads -- nothing like the 256 GiB
# the cache-first builder wanted, because nothing here goes through dask.
# ============================================================================

set -euo pipefail

module load miniforge/24.3.0-0

cd /orcd/home/002/codycruz/Ocean_Emulator

PROJECT_SITE_PACKAGES="/orcd/home/002/codycruz/Ocean_Emulator/.venv/lib/python3.11/site-packages"
export PYTHONPATH="/orcd/home/002/codycruz/Ocean_Emulator/src:${PROJECT_SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="/orcd/home/002/codycruz/Ocean_Emulator/.venv/bin/python"

# The reads are threaded inside one process; keep the maths libraries from
# oversubscribing on top of that.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# ---------------------------------------------------------------- sources
SOURCE_ZARR="${SOURCE_ZARR:-/orcd/data/abodner/003/LLC4320/LLC4320}"
MEANS_ZARR="${MEANS_ZARR:-/orcd/data/abodner/002/cody/LLC_means_stds/var_96_LLC_means.zarr}"
STDS_ZARR="${STDS_ZARR:-/orcd/data/abodner/002/cody/LLC_means_stds/var_96_LLC_stds.zarr}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/orcd/data/abodner/002/cody/LLC_patch/1104-4-tile-multicache}"

# ---------------------------------------------------------------- domain
# SPATIAL. JSON list of [face, i_start, i_end, j_start, j_end], or a path to a
# .json file holding one (use the file for a global tiling -- 176 entries do not
# belong on a command line). Every tile must share one shape.
#
# The 2x2 block of 1104 tiles with 24-cell overlap:
TILES="${TILES:-[[1,1068,2172,1068,2172],[1,2148,3252,1068,2172],[1,1068,2172,2148,3252],[1,2148,3252,2148,3252]]}"

# TEMPORAL. Same inclusive label slicing as the per-cache builder. Together
# these two windows cover every stored timestep: train ends 2012-09-13 23:00,
# val runs to the last sample at 2012-11-15 14:00. 8808 + 1503 = 10311.
TRAIN_START="${TRAIN_START:-2011-09-13}"
TRAIN_END="${TRAIN_END:-2012-09-13}"
VAL_START="${VAL_START:-2012-09-14}"
VAL_END="${VAL_END:-2012-11-15}"

# CHANNEL-WISE. Empty means the trainer's own PROGNOSTIC_VARS["all"] /
# BOUNDARY_VARS["all"], which is what keeps a cache loadable without config
# changes. Override with a JSON list to build a subset -- dropping W is the
# single biggest storage lever, at ~27% of the bytes for 20% of the channels.
#   PROGNOSTIC_CHANNELS='["Theta_0","Theta_1","Salt_0"]'
PROGNOSTIC_CHANNELS="${PROGNOSTIC_CHANNELS:-}"
BOUNDARY_CHANNELS="${BOUNDARY_CHANNELS:-}"

# ---------------------------------------------------------------- format
FLOAT_TYPE="${FLOAT_TYPE:-float16}"
TIME_CHUNK="${TIME_CHUNK:-1}"
COMPRESSOR="${COMPRESSOR:-lz4}"
COMPRESSION_LEVEL="${COMPRESSION_LEVEL:-5}"
SHUFFLE="${SHUFFLE:-shuffle}"
NAME_SUFFIX="${NAME_SUFFIX:-_trainval_ready}"

# ---------------------------------------------------------------- run mode
# init -> create the stores and their statics, write no timesteps
# fill -> write this array task's slice of the time axis into existing stores
# all  -> both, in one job (only sensible with TIME_SPLITS=1)
MODE="${MODE:-all}"
# How many equal time ranges the fill is divided into. Under `sbatch --array`
# this defaults to the array width, so the splits and the tasks cannot disagree
# -- a mismatch would silently leave timesteps unwritten or write some twice.
TIME_SPLITS="${TIME_SPLITS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
WORKERS="${WORKERS:-16}"
LOG_EVERY="${LOG_EVERY:-25}"
OVERWRITE="${OVERWRITE:-false}"
DRY_RUN="${DRY_RUN:-false}"

ARGS=(
  --source "${SOURCE_ZARR}"
  --means "${MEANS_ZARR}"
  --stds "${STDS_ZARR}"
  --output-root "${OUTPUT_ROOT}"
  --tiles "${TILES}"
  --name-suffix "${NAME_SUFFIX}"
  --train-start "${TRAIN_START}"
  --train-end "${TRAIN_END}"
  --val-start "${VAL_START}"
  --val-end "${VAL_END}"
  --float-type "${FLOAT_TYPE}"
  --time-chunk "${TIME_CHUNK}"
  --compressor "${COMPRESSOR}"
  --compression-level "${COMPRESSION_LEVEL}"
  --shuffle "${SHUFFLE}"
  --workers "${WORKERS}"
  --log-every "${LOG_EVERY}"
)

[[ -n "${PROGNOSTIC_CHANNELS}" ]] && ARGS+=(--prognostic-channels "${PROGNOSTIC_CHANNELS}")
[[ -n "${BOUNDARY_CHANNELS}" ]] && ARGS+=(--boundary-channels "${BOUNDARY_CHANNELS}")
[[ "${OVERWRITE}" == "true" ]] && ARGS+=(--overwrite)
[[ "${DRY_RUN}" == "true" ]] && ARGS+=(--dry-run)

case "${MODE}" in
  init) ARGS+=(--init) ;;
  fill)
    # A warning, not an error: rerunning a single failed task is
    # `--array=3` with an explicit TIME_SPLITS=8, where the two legitimately
    # differ. Only an accidental mismatch on a full submission is a problem.
    if [[ -n "${SLURM_ARRAY_TASK_COUNT:-}" && "${TIME_SPLITS}" != "${SLURM_ARRAY_TASK_COUNT}" ]]; then
      echo "NOTE: TIME_SPLITS=${TIME_SPLITS} but this array has ${SLURM_ARRAY_TASK_COUNT} task(s)."
      echo "      Fine when rerunning single tasks; otherwise part of the time axis goes unwritten."
    fi
    ARGS+=(--fill --time-splits "${TIME_SPLITS}"
           --time-split-index "${SLURM_ARRAY_TASK_ID:-0}")
    ;;
  all)
    # One job, start to finish. Nothing to split.
    TIME_SPLITS=1
    ;;
  *) echo "ERROR: MODE must be init, fill, or all (got '${MODE}')." >&2; exit 1 ;;
esac

echo "======== build multiple LLC packed caches (chunk-first) ========"
echo "mode=${MODE}  time_splits=${TIME_SPLITS}  array_task=${SLURM_ARRAY_TASK_ID:-<none>}"
echo "source=${SOURCE_ZARR}"
echo "output_root=${OUTPUT_ROOT}"
echo "tiles=${TILES}"
echo "train=[${TRAIN_START}:${TRAIN_END}], val=[${VAL_START}:${VAL_END}]"
echo "channels: prognostic=${PROGNOSTIC_CHANNELS:-<PROGNOSTIC_VARS[all]>} boundary=${BOUNDARY_CHANNELS:-<BOUNDARY_VARS[all]>}"
echo "format: ${FLOAT_TYPE}, ${COMPRESSOR}-${COMPRESSION_LEVEL} ${SHUFFLE}, time_chunk=${TIME_CHUNK}"
echo "workers=${WORKERS}, overwrite=${OVERWRITE}, dry_run=${DRY_RUN}"

"${PYTHON_BIN}" scripts/build_multiple_llc_patch_cache_compressed.py "${ARGS[@]}"
