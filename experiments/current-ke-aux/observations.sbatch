#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=current-ke-obs
#SBATCH --partition=grace
#SBATCH --account=ny_lz1955_multiscale
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=06:00:00
set -euo pipefail
export KE_ROOT=/projects/ny/lz1955/multiscale/jrusak/runs/current-ke-aux-20261010
export NVIDIA_VISIBLE_DEVICES=void
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
image=ghcr.io/m2lines/ocean-emulator-physicsnemo@sha256:7d78e8fb9592abdf046d88e55004f15cfe34f0c8ee38b17d83e98cc05b6f82c4
srun --container-env=NVIDIA_VISIBLE_DEVICES --container-image="$image" --container-mount-home --container-mounts="$KE_ROOT:$KE_ROOT,$KE_ROOT/eval-source/src:/workspace/src,/projects/ny/lz1955/multiscale/jrusak/data:/projects/ny/lz1955/multiscale/jrusak/data" bash -c '
set -euo pipefail
export PYTHONPATH=/workspace/src
exec /workspace/.venv/bin/python "$KE_ROOT/analysis-source/observations.py" --root "$KE_ROOT/evaluations-211962"
'
