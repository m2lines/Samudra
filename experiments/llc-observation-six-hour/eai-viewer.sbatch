#!/bin/bash -l
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=llc-viewer-export
#SBATCH --account=ny_lz1955_multiscale
#SBATCH --partition=beta
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=144
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=02:00:00
set -euo pipefail
root=${LLC_VIEWER_ROOT:?Set evaluation root}
run=${LLC_VIEWER_RUN:?Set frozen training root}
arm=${LLC_VIEWER_ARM:?Set arm}
mode=${1:-production}
[[ "$mode" == qualification || "$mode" == production ]]
on_exit() {
 rc=$?
 trap - EXIT
 callback="$root/$arm-$mode-callback-id"
 if [[ -f "$callback" ]]; then
  python3 "$HOME/.config/codex-followup/send.py" --host beta --event-id "$(cat "$callback")" || true
 fi
 exit "$rc"
}
trap on_exit EXIT
[[ -f "$root/forcing/COMPLETE.json" ]]
if [[ "$mode" == production ]]; then
 [[ -f "$root/qualification/COMPLETE.json" ]]
fi
image=ghcr.io/m2lines/ocean-emulator-physicsnemo@sha256:7d78e8fb9592abdf046d88e55004f15cfe34f0c8ee38b17d83e98cc05b6f82c4
scontrol show job "$SLURM_JOB_ID" > "$root/allocation-$SLURM_JOB_ID.txt"
srun --kill-on-bad-exit=1 --container-image="$image" --container-mount-home \
 --container-mounts="$run/code:/llc-code:ro,/projects/ny/lz1955/multiscale:/projects/ny/lz1955/multiscale" \
 bash "$root/scripts/run_viewer.sh" "$root" "$run" "$arm" "$mode"
