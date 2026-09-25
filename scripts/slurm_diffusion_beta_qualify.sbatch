#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=diffusion-beta-qualify
#SBATCH --account=ny_lz1955_multiscale
#SBATCH --partition=beta
#SBATCH --qos=test
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/home/jrusak/data/diffusion-interior-beta/logs/qualify-%j.out
#SBATCH --error=/mnt/home/jrusak/data/diffusion-interior-beta/logs/qualify-%j.err
set -euo pipefail
: "${CODE_COMMIT:?An immutable producer is required}"
root=/mnt/home/jrusak/data/diffusion-interior-beta
code="$root/code/$CODE_COMMIT"
test -f "$root/DATA_READY.json"
test -f "$root/checkpoints/selection.json"
export SAMUDRA_CODE_COMMIT="$CODE_COMMIT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
image=ghcr.io/m2lines/ocean-emulator-physicsnemo@sha256:7d78e8fb9592abdf046d88e55004f15cfe34f0c8ee38b17d83e98cc05b6f82c4 # pragma: allowlist secret
srun --ntasks=4 --gpus-per-task=1 --gpu-bind=single:1 --kill-on-bad-exit=1 \
  --container-image="$image" --container-mount-home \
  --container-mounts="$code:/pilot-code,$root:$root,/mnt/home/jrusak/data:/mnt/home/jrusak/data" \
  env PYTHONPATH=/pilot-code/src bash /pilot-code/scripts/diffusion_beta_qualify_rank.sh "$root"
