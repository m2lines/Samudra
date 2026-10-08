#!/bin/bash -l
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=diffusion-point-reference
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail
: "${CODE_COMMIT:?Immutable metric evaluator required}"
: "${BASELINE_CODE_COMMIT:?Immutable baseline model source required}"
root=/scratch/jr7309/diffusion-global-v1
code="$root/code/$CODE_COMMIT"
baseline="$root/code/$BASELINE_CODE_COMMIT"
comparison=/scratch/jr7309/runs/2026-09-29-observation-global
image=/scratch/jr7309/.apptainer-images/ghcr.io-m2lines-ocean-emulator-physicsnemo-26.05-9cd36b1bdcf4921fb027494afa4157e217818ccd.sif
export SAMUDRA_CODE_COMMIT="$CODE_COMMIT"
export APPTAINER_CACHEDIR=/scratch/jr7309/apptainer-cache
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 WANDB_MODE=disabled
srun apptainer exec --nv --bind "$code:/pilot-code,$baseline:/baseline-code,/scratch:/scratch" "$image" \
 env PYTHONPATH=/baseline-code/src /workspace/.venv/bin/python \
 /pilot-code/scripts/evaluate_diffusion_deterministic_reference.py \
 --run "$comparison/conditioned-mixed-global" --data "$root/data/observations" \
 --previous "$comparison/global-endpoint-monthly" \
 --metric-helper /pilot-code/src/samudra/experiments/diffusion_calibration.py \
 --output "$root/evaluation/deterministic-point-reference"
