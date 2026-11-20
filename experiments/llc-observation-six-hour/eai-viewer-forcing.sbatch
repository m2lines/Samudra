#!/bin/bash -l
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=llc-viewer-forcing
#SBATCH --account=ny_lz1955_multiscale
#SBATCH --partition=grace
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -euo pipefail
root=${LLC_VIEWER_ROOT:?Set evaluation root}
parent=/projects/ny/lz1955/multiscale/jrusak/runs/llc-obs-six-hour-20261120
base=/projects/ny/lz1955/multiscale/jrusak/data
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$parent/prep-code/src:$parent/prep-code/data"
on_exit() {
 rc=$?
 trap - EXIT
 if [[ -f "$root/preparation-callback-id" ]]; then
  python3 "$HOME/.config/codex-followup/send.py" --host alpha --event-id "$(cat "$root/preparation-callback-id")" || true
 fi
 exit "$rc"
}
trap on_exit EXIT
"$parent/prep-env/bin/python" "$root/scripts/prepare_viewer_forcing.py" \
 --prepared "$base/llc_obs_six_hour_v1" --source "$base/obs_full_range/prepared" \
 --source-grid "$parent/era5-source-grid.nc" --output "$root/forcing"
