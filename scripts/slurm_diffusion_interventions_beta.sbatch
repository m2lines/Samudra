#!/bin/bash -l
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=diffusion-interventions
#SBATCH --account=ny_lz1955_multiscale
#SBATCH --partition=beta
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=144
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=16:00:00
set -euo pipefail
: "${CODE_ID:?Immutable source archive required}"
root=${CAMPAIGN_ROOT:-/projects/ny/lz1955/multiscale/jrusak/runs/diffusion-interventions-v1}
code="$root/code/$CODE_ID"
export SAMUDRA_CODE_COMMIT="$CODE_ID"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 TORCHINDUCTOR_COMPILE_THREADS=2
export WANDB_MODE=online
image=ghcr.io/m2lines/ocean-emulator-physicsnemo@sha256:7d78e8fb9592abdf046d88e55004f15cfe34f0c8ee38b17d83e98cc05b6f82c4 # pragma: allowlist secret
args=(--root "$root" --campaign "${CAMPAIGN:-v1}")
if [[ "${SMOKE:-0}" == 1 ]]; then args+=(--smoke); export WANDB_MODE=disabled; fi
scontrol show job "$SLURM_JOB_ID" > "$root/logs/allocation-$SLURM_JOB_ID.txt"
srun --kill-on-bad-exit=1 --container-image="$image" --container-mount-home \
  --container-mounts="$code:/pilot-code,$root/native:/pilot-native,/projects/ny/lz1955/multiscale:/projects/ny/lz1955/multiscale" \
  env CC=/usr/bin/gcc CXX=/usr/bin/g++ PYTHONPATH=/pilot-code/src:/pilot-native \
  /workspace/.venv/bin/python /pilot-code/scripts/run_diffusion_interventions.py "${args[@]}"
