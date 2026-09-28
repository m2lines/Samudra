#!/bin/bash -l
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=diffusion-latent
#SBATCH --account=mit_general
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/orcd/pool/008/jrusak/diffusion-interior-engaging/logs/latent-%A_%a.out
#SBATCH --error=/orcd/pool/008/jrusak/diffusion-interior-engaging/logs/latent-%A_%a.err
set -euo pipefail
module load apptainer/1.5.2
: "${CODE_COMMIT:?Immutable producer required}"
: "${RUN_GROUP:?Run group required}"
: "${TRAIN_PHASE:?qualify, om4 or observation required}"
root=/orcd/pool/008/jrusak/diffusion-interior-engaging
code="$root/code/$CODE_COMMIT"
image=/orcd/scratch/orcd/014/jrusak/apptainer_cache/images/diffusion-26.05-9cd36b1-amd64.sif
seed=$((1729 + ${SLURM_ARRAY_TASK_ID:-0}))
extra=()
output="$root/runs/$RUN_GROUP/D-$seed/$TRAIN_PHASE"
if [[ "$TRAIN_PHASE" != qualify ]]; then
  : "${QUALIFICATION_JOB:?Real-grid qualification required}"
  : "${TRAIN_UPDATES:?Update ceiling required}"
  : "${TRAIN_HOURS:?Time ceiling required}"
  extra+=(--qualification "$root/runs/$RUN_GROUP/qualification-$QUALIFICATION_JOB/QUALIFIED.json" --updates "$TRAIN_UPDATES" --hours "$TRAIN_HOURS")
else
  output="$root/runs/$RUN_GROUP/qualification-$SLURM_JOB_ID"
fi
if [[ "$TRAIN_PHASE" == observation ]]; then
  extra+=(--pretrained "$root/runs/$RUN_GROUP/D-$seed/om4/best.pt")
fi
export SAMUDRA_CODE_COMMIT="$CODE_COMMIT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
srun --gpus-per-task=1 apptainer exec --nv --bind "$code:/pilot-code,/orcd:/orcd" "$image" \
  env PYTHONPATH=/pilot-code/src /workspace/.venv/bin/python -m samudra.experiments.diffusion_latent_train \
  --root "$root" --output "$output" --phase "$TRAIN_PHASE" --seed "$seed" \
  --decoder-width 192 --latent-width 128 --processor-depth 4 --sampling-steps 16 \
  "${extra[@]}"
