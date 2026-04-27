#!/bin/bash
# Slurm wrapper for `scripts/migrate_datasets_to_pubs.sh`.
#
# Streams the configured FOMO datasets from `nyu-osn:emulators` to
# `nyu-osn-public:m2lines-pubs/FOMO` via rclone running on a torch CPU node.
# (Server-side S3 copy is not available because the source and destination
# buckets use different keys, so the data flows through the compute node.)
#
# Prereqs on torch:
#   - `rclone` on PATH (e.g. ~/.local/bin/rclone), v1.69+ recommended.
#   - `~/.config/rclone/rclone.conf` contains both `[nyu-osn]` (read on
#     `emulators`) and `[nyu-osn-public]` (write on `m2lines-pubs`), each with
#     `no_check_bucket = true`.
#
# Submit from the repo root:
#   sbatch scripts/migrate_datasets_to_pubs.sbatch
#
# Optional env vars:
#   EXTRA_ARGS   extra rclone args (default: "--transfers 32 --checkers 32")
#   DRY_RUN=1    forward --dry-run to rclone

#SBATCH --job-name=oe-migrate
#SBATCH --output=slurm-%j.out
#SBATCH --account=torch_pr_347_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail

SCRIPT="${SLURM_SUBMIT_DIR:-$PWD}/scripts/migrate_datasets_to_pubs.sh"
if [ ! -x "$SCRIPT" ]; then
    echo "Error: $SCRIPT not found or not executable" >&2
    exit 1
fi

EXTRA_ARGS="${EXTRA_ARGS:---transfers 32 --checkers 32}"
DRY_FLAG=()
if [ "${DRY_RUN:-0}" = "1" ]; then
    DRY_FLAG=(--dry-run)
fi

echo "Host:        $(hostname)"
echo "Job ID:      ${SLURM_JOB_ID:-N/A}"
echo "Script:      $SCRIPT"
echo "Extra args:  $EXTRA_ARGS"
echo "Dry run:     ${DRY_RUN:-0}"
echo ""

exec "$SCRIPT" "${DRY_FLAG[@]}" -- $EXTRA_ARGS
