#!/bin/bash -l
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
# Usage: SPIKE_RUN_ROOT=... SPIKE_CODE=... sbatch replay.sbatch <window> <arm>
# Same container, node shape and world size as #1037's long runs (four GB200s);
# exact replay needs all three, since checkpoints carry four ranks' RNG state.
#SBATCH --job-name=spike-replay
#SBATCH --account=ny_lz1955_multiscale
#SBATCH --partition=beta
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=144
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=02:30:00
set -euo pipefail
: "${SPIKE_RUN_ROOT:?}" "${SPIKE_CODE:?Extracted git archive of the replay commit}"
mkdir -p "$SPIKE_RUN_ROOT/logs"
scontrol show job "$SLURM_JOB_ID" > "$SPIKE_RUN_ROOT/logs/allocation-$SLURM_JOB_ID.txt"
image=ghcr.io/m2lines/ocean-emulator-physicsnemo@sha256:7d78e8fb9592abdf046d88e55004f15cfe34f0c8ee38b17d83e98cc05b6f82c4
srun --kill-on-bad-exit=1 --container-image="$image" --container-mount-home \
  --container-mounts="$SPIKE_CODE:/spike-code:ro,/projects/ny/lz1955/multiscale:/projects/ny/lz1955/multiscale" \
  bash /spike-code/experiments/llc-obs-spike-diagnosis/run_replay.sh "$1" "$2"
