#!/bin/bash -l
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=llc-obs-six-hour
#SBATCH --account=ny_lz1955_multiscale
#SBATCH --partition=beta
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=144
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=06:00:00
set -euo pipefail
export LLC_OBS_RUN_ROOT=${LLC_OBS_RUN_ROOT:-/projects/ny/lz1955/multiscale/jrusak/runs/llc-obs-six-hour-20261120}
mode=${1:-production}
on_exit() {
 rc=$?
 trap - EXIT
 callback="$LLC_OBS_RUN_ROOT/${mode}-callback-id"
 if [[ -f "$callback" ]]; then
  python3 "$HOME/.config/codex-followup/send.py" --host beta --event-id "$(cat "$callback")" || true
 fi
 exit "$rc"
}
trap on_exit EXIT
if [[ -z ${WANDB_API_KEY:-} ]]; then
 export WANDB_API_KEY="$(cat "$HOME/.config/wandb/llc-obs-api-key")"
fi
scontrol show job "$SLURM_JOB_ID" > "$LLC_OBS_RUN_ROOT/allocation-$SLURM_JOB_ID.txt"
image=ghcr.io/m2lines/ocean-emulator-physicsnemo@sha256:7d78e8fb9592abdf046d88e55004f15cfe34f0c8ee38b17d83e98cc05b6f82c4
srun --kill-on-bad-exit=1 --container-image="$image" --container-mount-home \
 --container-mounts="$LLC_OBS_RUN_ROOT/code:/llc-code:ro,/projects/ny/lz1955/multiscale:/projects/ny/lz1955/multiscale" \
 bash /llc-code/experiments/llc-observation-six-hour/run_eai.sh "$mode"
