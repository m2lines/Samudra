#!/bin/bash -l
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=diff-coast-probe
#SBATCH --account=ny_lz1955_multiscale
#SBATCH --partition=beta_test
#SBATCH --qos=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --mem=192G
#SBATCH --time=01:30:00
set -euo pipefail
: "${CODE_ID:?Immutable diagnostic source required}"
root=/projects/ny/lz1955/multiscale/jrusak/runs/diffusion-interventions-v2
code="$root/code/$CODE_ID"
output="$root/diagnostics/coastal-decoder-v1"
export SAMUDRA_CODE_COMMIT="$CODE_ID"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
image=ghcr.io/m2lines/ocean-emulator-physicsnemo@sha256:7d78e8fb9592abdf046d88e55004f15cfe34f0c8ee38b17d83e98cc05b6f82c4 # pragma: allowlist secret
mkdir -p "$output"
scontrol show job "$SLURM_JOB_ID" > "$output/allocation-$SLURM_JOB_ID.txt"
srun --kill-on-bad-exit=1 --container-image="$image" --container-mount-home \
  --container-mounts="$code:/probe-code,$root/native:/probe-native,/projects/ny/lz1955/multiscale:/projects/ny/lz1955/multiscale" \
  env CC=/usr/bin/gcc CXX=/usr/bin/g++ PYTHONPATH=/probe-code/src:/probe-native \
  /workspace/.venv/bin/python /probe-code/scripts/probe_diffusion_coastal_decoder.py \
  --root "$root" --output "$output" --land-fill
