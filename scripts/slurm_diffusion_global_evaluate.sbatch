#!/bin/bash -l
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=diffusion-global-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail
: "${CODE_COMMIT:?Immutable evaluator required}"
: "${EVAL_CHECKPOINT:?Immutable checkpoint required}"
: "${EVAL_NAME:?Unique evaluation name required}"
module load apptainer/1.5.2
root=/orcd/pool/008/jrusak/diffusion-global-v1
code="$root/code/$CODE_COMMIT"
image=/orcd/scratch/orcd/014/jrusak/apptainer_cache/images/diffusion-26.05-9cd36b1-amd64.sif
export SAMUDRA_CODE_COMMIT="$CODE_COMMIT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export WANDB_MODE=disabled
test -f "$EVAL_CHECKPOINT"
srun apptainer exec --nv --bind "$code:/pilot-code,/orcd:/orcd" "$image" \
 env CC=/usr/bin/gcc CXX=/usr/bin/g++ PYTHONPATH=/pilot-code/src /workspace/.venv/bin/python \
 -m samudra.experiments.diffusion_global_evaluate \
 --data "$root/data/observations" --checkpoint "$EVAL_CHECKPOINT" \
 --reference "$root/selection-reference.json" --output "$root/evaluation/$EVAL_NAME" \
 --steps "${EVAL_STEPS:-32}" --split "${EVAL_SPLIT:-validation}"
