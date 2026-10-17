#!/bin/bash -l
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=monthly-latent-pilots
#SBATCH --account=ny_lz1955_multiscale
#SBATCH --partition=beta
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=144
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=24:00:00
set -euo pipefail
: "${PILOT_ROOT:?}"
: "${PRODUCER:?}"
: "${QUALIFICATION:?}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_MODE=online
code="$PILOT_ROOT/code/$PRODUCER"
image=ghcr.io/m2lines/ocean-emulator-physicsnemo@sha256:7d78e8fb9592abdf046d88e55004f15cfe34f0c8ee38b17d83e98cc05b6f82c4 # pragma: allowlist secret
scontrol show job "$SLURM_JOB_ID" > "$PILOT_ROOT/logs/allocation-$SLURM_JOB_ID.txt"
srun --cpus-per-task="$SLURM_CPUS_PER_TASK" --kill-on-bad-exit=1 --container-image="$image" --container-mount-home \
 --container-mounts="$code:/pilot-code:ro,/projects/ny/lz1955/multiscale:/projects/ny/lz1955/multiscale" \
 env PYTHONPATH=/pilot-code/src /workspace/.venv/bin/python /pilot-code/scripts/launch_monthly_latent_pilots.py \
 --root "$PILOT_ROOT" --producer "$PRODUCER" --qualification "$QUALIFICATION"
