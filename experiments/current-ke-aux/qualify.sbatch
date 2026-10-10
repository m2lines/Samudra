#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --partition=beta_test
#SBATCH --qos=test
#SBATCH --account=ny_lz1955_multiscale
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=00:30:00
set -euo pipefail
export KE_ROOT=/projects/ny/lz1955/multiscale/jrusak/runs/current-ke-aux-20261010
export KE_DATA=/projects/ny/lz1955/multiscale/jrusak/data/om4/v2026-09
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 WANDB_MODE=disabled
export SAMUDRA_CODE_COMMIT=$(cat "$KE_ROOT/commit.txt")
export SAMUDRA_CONTAINER_IMAGE_REF=ghcr.io/m2lines/ocean-emulator-physicsnemo@sha256:7d78e8fb9592abdf046d88e55004f15cfe34f0c8ee38b17d83e98cc05b6f82c4
srun --container-image="$SAMUDRA_CONTAINER_IMAGE_REF" --container-mount-home --container-mounts="$KE_ROOT/source/src:/workspace/src,$KE_ROOT/native:/workspace/native,$KE_ROOT:$KE_ROOT,$KE_DATA:$KE_DATA,/mnt/home/jrusak/data/diffusion-interior-beta/data/om4_onedeg_v3:/mnt/home/jrusak/data/diffusion-interior-beta/data/om4_onedeg_v3,$KE_ROOT/cache:/workspace/.data_cache" bash -c '
set -euo pipefail
export PYTHONPATH=/workspace/src:/workspace/native
cd "$KE_ROOT/source"
if [ ! -f "$KE_ROOT/targets/READY.json" ]; then
 /workspace/.venv/bin/python experiments/current-ke-aux/prepare_targets.py --parent "$KE_DATA/om4_quarterdeg_surface_variance_1deg" --coarse "$KE_DATA/om4_onedeg" --output "$KE_ROOT/targets"
fi
for mode in aligned seasonal; do
 /workspace/.venv/bin/python experiments/current-ke-aux/qualify.py "$KE_ROOT/$mode-train.yaml" --output "$KE_ROOT/qualification-$mode.json"
done
'
