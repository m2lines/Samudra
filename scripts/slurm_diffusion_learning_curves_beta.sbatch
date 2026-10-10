#!/bin/bash -l
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=diffint-v2-curves
#SBATCH --account=ny_lz1955_multiscale
#SBATCH --partition=beta
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=144
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=07:00:00
set -euo pipefail
: "${CODE_ID:?Immutable evaluation source required}"
root=/projects/ny/lz1955/multiscale/jrusak/runs/diffusion-interventions-v2
code="$root/code/$CODE_ID"
export SAMUDRA_CODE_COMMIT="$CODE_ID"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
image=ghcr.io/m2lines/ocean-emulator-physicsnemo@sha256:7d78e8fb9592abdf046d88e55004f15cfe34f0c8ee38b17d83e98cc05b6f82c4 # pragma: allowlist secret
scontrol show job "$SLURM_JOB_ID" > "$root/logs/allocation-$SLURM_JOB_ID.txt"
srun --kill-on-bad-exit=1 --container-image="$image" --container-mount-home \
  --container-mounts="$code:/curve-code,$root/native:/curve-native,/projects/ny/lz1955/multiscale:/projects/ny/lz1955/multiscale" \
  env PYTHONPATH=/curve-code/src:/curve-native \
  /workspace/.venv/bin/python /curve-code/scripts/run_diffusion_learning_curves.py --root "$root"
