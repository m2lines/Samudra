#!/bin/bash -l
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=diffusion-hardware
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail
: "${GPU_NAME:?Hardware name required}"
root=/scratch/jr7309/diffusion-global-v1
code="$root/code/4ecba36a40b7b581edeaa2292c90d1953fca7aa0"
work="$root/hardware-$GPU_NAME"
image=/scratch/jr7309/.apptainer-images/ghcr.io-m2lines-ocean-emulator-physicsnemo-26.05-9cd36b1bdcf4921fb027494afa4157e217818ccd.sif
export SAMUDRA_CODE_COMMIT=4ecba36a40b7b581edeaa2292c90d1953fca7aa0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export TORCHINDUCTOR_COMPILE_THREADS=2
export TORCHINDUCTOR_CACHE_DIR="$work/inductor-cache"
export TRITON_CACHE_DIR="$work/triton-cache"
export APPTAINER_CACHEDIR=/scratch/jr7309/apptainer-cache
export WANDB_MODE=disabled
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
srun apptainer exec --nv --bind "$code:/pilot-code,/scratch:/scratch" "$image" \
 env CC=/usr/bin/gcc CXX=/usr/bin/g++ PYTHONPATH=/pilot-code/src /workspace/.venv/bin/python \
 -m samudra.experiments.diffusion_global_train --root "$work" --output "$work/qualification" \
 --reference /scratch/jr7309/runs/2026-09-29-observation-global/fit-conditioned/selection-reference.json \
 --qualify --compile-decoder
