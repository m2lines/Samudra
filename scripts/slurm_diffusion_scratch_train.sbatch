#!/bin/bash -l
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=diffusion-scratch
#SBATCH --account=mit_general
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-1%2
#SBATCH --requeue
#SBATCH --output=/orcd/pool/008/jrusak/diffusion-interior-engaging/logs/scratch-%A_%a.out
#SBATCH --error=/orcd/pool/008/jrusak/diffusion-interior-engaging/logs/scratch-%A_%a.err
set -euo pipefail
module load apptainer/1.5.2
: "${CODE_COMMIT:?Immutable qualified producer required}"
: "${QUALIFICATION_JOB:?Scratch GPU qualification required}"
: "${TRAIN_PHASE:?om4 or observation required}"
: "${TRAIN_UPDATES:?Update ceiling required}"
: "${TRAIN_HOURS:?Time ceiling required}"
: "${RUN_GROUP:?Separate scratch run group required}"
[[ "$TRAIN_PHASE" == om4 || "$TRAIN_PHASE" == observation ]]
root=/orcd/pool/008/jrusak/diffusion-interior-engaging
code="$root/code/$CODE_COMMIT"
image=/orcd/scratch/orcd/014/jrusak/apptainer_cache/images/diffusion-26.05-9cd36b1-amd64.sif
seed=$((1729 + SLURM_ARRAY_TASK_ID))
[[ "$seed" == 1729 || "$seed" == 1730 ]]
run="$root/runs/$RUN_GROUP/B-$seed"
extra=()
if [[ "$TRAIN_PHASE" == observation ]]; then
  extra+=(--pretrained "$run/om4/best.pt")
fi
export SAMUDRA_CODE_COMMIT="$CODE_COMMIT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
srun --gpus-per-task=1 apptainer exec --nv --bind "$code:/pilot-code,/orcd:/orcd" "$image" \
  env PYTHONPATH=/pilot-code/src /workspace/.venv/bin/python -m samudra.experiments.diffusion_train \
  --root "$root" --arm B --initialization scratch --decoder-width 192 --phase "$TRAIN_PHASE" \
  --qualification "$root/runs/scratch-qualification/$QUALIFICATION_JOB/QUALIFIED.json" \
  --output "$run/$TRAIN_PHASE" --name "$RUN_GROUP-B-$seed-$TRAIN_PHASE" --seed "$seed" \
  --updates "$TRAIN_UPDATES" --hours "$TRAIN_HOURS" --readers 4 --batch-size 1 \
  --checkpoint-every 100 --validate-every 500 --wandb-mode offline "${extra[@]}"
