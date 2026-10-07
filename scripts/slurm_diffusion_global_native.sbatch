#!/bin/bash -l
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=diffusion-global-native
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail
: "${CODE_COMMIT:?Immutable evaluator required}"
: "${EVAL_CHECKPOINT:?Final checkpoint required}"
: "${EVAL_NAME:?Unique evaluation name required}"
root=/scratch/jr7309/diffusion-global-v1
code="$root/code/$CODE_COMMIT"
image=/scratch/jr7309/.apptainer-images/ghcr.io-m2lines-ocean-emulator-physicsnemo-26.05-9cd36b1bdcf4921fb027494afa4157e217818ccd.sif
export SAMUDRA_CODE_COMMIT="$CODE_COMMIT"
export APPTAINER_CACHEDIR=/scratch/jr7309/apptainer-cache
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 WANDB_MODE=disabled
srun apptainer exec --nv --bind "$code:/pilot-code,/scratch:/scratch" "$image" \
 env CC=/usr/bin/gcc CXX=/usr/bin/g++ PYTHONPATH=/pilot-code/src /workspace/.venv/bin/python \
 -m samudra.experiments.diffusion_latent_native \
 --root "$root" --checkpoint "$EVAL_CHECKPOINT" \
 --output "$root/evaluation/$EVAL_NAME" --global-final
