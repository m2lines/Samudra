#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=obs-D-beta-qualify
#SBATCH --account=ny_lz1955_multiscale
#SBATCH --partition=beta
#SBATCH --qos=test
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
set -euo pipefail
root=/mnt/home/jrusak/data/obs_full_range/d-observation-pilot
code="$root/code/beta-source-d894"
output="$root/beta-run/source-contract"
mkdir -p "$output"
export PATH="$HOME/bin:$PATH" RCLONE_S3_NO_CHECK_BUCKET=true
rclone copyto nyu-osn:emulators/jr7309/data/observation_d_pilot/2026-09-22-checkpoints/D-source.pt "$root/beta-run/D-source.pt"
echo '22e629f41c418fa93ee3ec0a3257d80d71d77b02b2fe1cbd89eae04507d22905  D-source.pt' | (cd "$root/beta-run" && sha256sum --check)
(cd "$root/samples" && sha256sum --check --status SHA256SUMS)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export SAMUDRA_CODE_COMMIT=d8949bb15d9fa92e6c8c94702092360bbe9005ef
image=ghcr.io/m2lines/ocean-emulator-physicsnemo@sha256:7d78e8fb9592abdf046d88e55004f15cfe34f0c8ee38b17d83e98cc05b6f82c4 # pragma: allowlist secret
srun --nodes=1 --ntasks=1 --container-image="$image" --container-mount-home \
  --container-mounts="$code:/pilot-code,$root:$root" \
  env PYTHONPATH=/pilot-code/src CUDA_VISIBLE_DEVICES=0 \
  /workspace/.venv/bin/python -m samudra.experiments.observation_model \
  --checkpoint="$root/beta-run/D-source.pt" --grid="$root/samples/grid.npz" --output="$output"
