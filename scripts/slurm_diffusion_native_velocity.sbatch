#!/bin/bash -l
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=diffusion-native-velocity
#SBATCH --account=mit_general
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-1%2
#SBATCH --requeue
#SBATCH --output=/orcd/pool/008/jrusak/diffusion-interior-engaging/logs/scratch-velocity-%A_%a.out
#SBATCH --error=/orcd/pool/008/jrusak/diffusion-interior-engaging/logs/scratch-velocity-%A_%a.err
set -euo pipefail
module load apptainer/1.5.2
: "${CODE_COMMIT:?Immutable evaluator required}"
: "${RUN_GROUP:?Scratch run group required}"
: "${CHECKPOINT_PHASE:?om4 or observation required}"
[[ "$CHECKPOINT_PHASE" == om4 || "$CHECKPOINT_PHASE" == observation ]]
root=/orcd/pool/008/jrusak/diffusion-interior-engaging
code="$root/code/$CODE_COMMIT"
image=/orcd/scratch/orcd/014/jrusak/apptainer_cache/images/diffusion-26.05-9cd36b1-amd64.sif
seed=$((1729 + SLURM_ARRAY_TASK_ID))
[[ "$seed" == 1729 || "$seed" == 1730 ]]
run="$root/runs/$RUN_GROUP/B-$seed"
export SAMUDRA_CODE_COMMIT="$CODE_COMMIT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
execute() {
  srun --gpus-per-task=1 apptainer exec --nv --bind "$code:/pilot-code,/orcd:/orcd" "$image" \
    env PYTHONPATH=/pilot-code/src /workspace/.venv/bin/python -m "$@"
}
execute samudra.experiments.diffusion_native_controls --root "$root" \
  --checkpoint "$run/$CHECKPOINT_PHASE/best.pt" \
  --output "$run/native-velocity-v1/$CHECKPOINT_PHASE" --velocity-diagnostics
