#!/bin/bash
#SBATCH -p mit_normal
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --job-name=finalize_fw_720tile
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --time=00:20:00
#SBATCH -o /orcd/home/002/codycruz/Ocean_Emulator/logs/%x-%j.out
#SBATCH -e /orcd/home/002/codycruz/Ocean_Emulator/logs/%x-%j.out

# ============================================================================
# Swap the staged 5-channel boundary arrays into place on a packed LLC cache.
#
# This is the third phase of scripts/add_boundary_channels.py. --init and --fill
# leave the live arrays untouched, so the long write can run under a training
# job; only this step changes what a reader sees. It is a directory rename plus
# a metadata rewrite -- seconds, no compute, hence the tiny allocation.
#
# ONLY RUN THIS WHEN NOTHING IS MID-EPOCH ON THE TARGET. A job that has the
# store open across the swap can read a 4-wide array against 5-wide metadata.
#
# The old arrays are moved to a <cache>.boundary_v1_backup sibling rather than
# deleted, so the swap is reversible. That suffix is not ".zarr", so a
# data_location pointed at a DIRECTORY of caches will not pick the backup up as
# an extra tile.
#
#   sbatch --dependency=afterok:<fill_jobid> JOBS/other/finalize_boundary_channels.sh
#
# Knobs (all overridable from the environment):
#   SOURCE_CACHE  packed cache to finalize
#   EXTRA         channel names added by --init/--fill (must match that run)
# ============================================================================

set -euo pipefail

REPO=/orcd/home/002/codycruz/Ocean_Emulator
cd "${REPO}"

SOURCE_CACHE="${SOURCE_CACHE:-/orcd/data/abodner/002/cody/LLC_patch/LLC4320_face1_i2880-3600_j720-1440_trainval_ready_20110913_20121014_t1.zarr}"
EXTRA="${EXTRA:-oceFWflx}"

echo "finalizing ${SOURCE_CACHE}"
echo "  adding channels: ${EXTRA}"
echo "  boundary before: $(.venv/bin/python -c "
import json, zarr, sys
print(json.loads(zarr.open(sys.argv[1], mode='r').attrs['boundary_channel_names_json']))
" "${SOURCE_CACHE}")"

.venv/bin/python scripts/add_boundary_channels.py \
  --source-cache "${SOURCE_CACHE}" \
  --in-place \
  --finalize \
  --extra ${EXTRA}

echo "  boundary after:  $(.venv/bin/python -c "
import json, zarr, sys
print(json.loads(zarr.open(sys.argv[1], mode='r').attrs['boundary_channel_names_json']))
" "${SOURCE_CACHE}")"
