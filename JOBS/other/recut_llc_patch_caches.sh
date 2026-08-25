#!/bin/bash
#SBATCH -p mit_normal
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --job-name=2026-08-25-llc_recut_752
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=12:00:00
#SBATCH -o /orcd/home/002/codycruz/Ocean_Emulator/logs/%x-%A_%a-%j.out
#SBATCH -e /orcd/home/002/codycruz/Ocean_Emulator/logs/%x-%A_%a-%j.out

# ============================================================================
# Re-cut the 1104x1104 caches into 752x752 tiles (720 base + 16 halo per side).
#
# This reads the EXISTING caches, not the raw store, because every cell of the
# new tiling is already in them and doing so is ~5x cheaper:
#     from the caches    read  13 TB
#     from the raw store read  66 TB
# The one exception is oceFWflx, which the 1104 caches predate; that comes from
# the raw store in the same pass (one globe chunk per timestep).
#
# Each 752 tile straddles the 1104 seams, so it is gathered from all four source
# caches -- a ~748x748 bulk plus 28-wide slivers. The script refuses to build a
# tile the sources cannot completely fill rather than writing zeros.
#
# Output: ~4.9 TB for the four tiles (205 channels, W dropped).
#
# TWO-STEP SUBMISSION, as for the multicache builder:
#     init=$(MODE=init sbatch --parsable JOBS/other/recut_llc_patch_caches.sh)
#     MODE=fill sbatch --dependency=afterok:${init} --array=0-7 \
#         JOBS/other/recut_llc_patch_caches.sh
#
# Short smoke test on one job:
#     MODE=all TIME_STOP=8 OVERWRITE=true bash JOBS/other/recut_llc_patch_caches.sh
# ============================================================================

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

SOURCE_ROOT="${SOURCE_ROOT:-/orcd/data/abodner/002/cody/LLC_patch/1104-4-tile-multicache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/orcd/data/abodner/002/cody/LLC_patch/752-4-tile}"
RAW_STORE="${RAW_STORE:-/orcd/data/abodner/003/LLC4320/LLC4320}"
MEANS_ZARR="${MEANS_ZARR:-/orcd/data/abodner/002/cody/LLC_means_stds/var_96_LLC_means.zarr}"
STDS_ZARR="${STDS_ZARR:-/orcd/data/abodner/002/cody/LLC_means_stds/var_96_LLC_stds.zarr}"

# FULL extents, halo included. Base 720 tiles are chunk-aligned on the LLC grid
# (1440/2160/2880 are multiples of 720); +/-16 makes them 752, divisible by 16.
#   T11 i[1424:2176) j[1424:2176)      T21 i[2144:2896) j[1424:2176)
#   T12 i[1424:2176) j[2144:2896)      T22 i[2144:2896) j[2144:2896)
TILES="${TILES:-[[1,1424,2176,1424,2176],[1,2144,2896,1424,2176],[1,1424,2176,2144,2896],[1,2144,2896,2144,2896]]}"
HALO="${HALO:-16}"

# Prognostic variables to leave out, by base name.
DROP_CHANNELS="${DROP_CHANNELS-W}"
# Boundary variables to append from the raw store. oceFWflx is the salinity
# counterpart of oceQnet here; oceSflux is sea-ice-only and identically zero in
# this patch, so it would be a dead channel with std 0.
# Its mean/std must already be in MEANS_ZARR/STDS_ZARR:
#     uv run notebooks/LLC_add_mean_std.py --vars oceFWflx
EXTRA_BOUNDARY="${EXTRA_BOUNDARY-oceFWflx}"

TIME_CHUNK="${TIME_CHUNK:-1}"
COMPRESSOR="${COMPRESSOR:-lz4}"
COMPRESSION_LEVEL="${COMPRESSION_LEVEL:-5}"
SHUFFLE="${SHUFFLE:-shuffle}"
NAME_SUFFIX="${NAME_SUFFIX:-_trainval_ready}"

MODE="${MODE:-all}"
TIME_SPLITS="${TIME_SPLITS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
TIME_STOP="${TIME_STOP:-}"          # smoke tests: stop after this many timesteps
LOG_EVERY="${LOG_EVERY:-25}"
OVERWRITE="${OVERWRITE:-false}"
DRY_RUN="${DRY_RUN:-false}"

ARGS=(
  --source-root "${SOURCE_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --raw-store "${RAW_STORE}"
  --means "${MEANS_ZARR}"
  --stds "${STDS_ZARR}"
  --tiles "${TILES}"
  --halo "${HALO}"
  --name-suffix "${NAME_SUFFIX}"
  --time-chunk "${TIME_CHUNK}"
  --compressor "${COMPRESSOR}"
  --compression-level "${COMPRESSION_LEVEL}"
  --shuffle "${SHUFFLE}"
  --log-every "${LOG_EVERY}"
)
# shellcheck disable=SC2206  # word splitting is how multiple names arrive
[[ -n "${DROP_CHANNELS}" ]] && ARGS+=(--drop-channels ${DROP_CHANNELS})
# shellcheck disable=SC2206
[[ -n "${EXTRA_BOUNDARY}" ]] && ARGS+=(--extra-boundary ${EXTRA_BOUNDARY})
[[ "${OVERWRITE}" == "true" ]] && ARGS+=(--overwrite)
[[ "${DRY_RUN}" == "true" ]] && ARGS+=(--dry-run)
[[ -n "${TIME_STOP}" ]] && ARGS+=(--time-index-stop "${TIME_STOP}")

case "${MODE}" in
  init) ARGS+=(--init) ;;
  fill)
    ARGS+=(--fill --time-splits "${TIME_SPLITS}"
           --time-split-index "${SLURM_ARRAY_TASK_ID:-0}")
    ;;
  all)  TIME_SPLITS=1 ;;
  *) echo "ERROR: MODE must be init, fill, or all (got '${MODE}')." >&2; exit 1 ;;
esac

echo "======== re-cut LLC packed caches ========"
echo "mode=${MODE} time_splits=${TIME_SPLITS} array_task=${SLURM_ARRAY_TASK_ID:-<none>}"
echo "source_root=${SOURCE_ROOT}"
echo "output_root=${OUTPUT_ROOT}"
echo "tiles=${TILES} (halo=${HALO})"
echo "drop=${DROP_CHANNELS:-<nothing>} extra_boundary=${EXTRA_BOUNDARY:-<none>} (from ${RAW_STORE})"
echo "time_stop=${TIME_STOP:-<all>} overwrite=${OVERWRITE} dry_run=${DRY_RUN}"

"${PYTHON_BIN}" scripts/recut_llc_patch_caches.py "${ARGS[@]}"
