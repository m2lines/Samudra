#!/bin/bash -l
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=latent-report
#SBATCH --account=mit_general
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=/orcd/pool/008/jrusak/diffusion-interior-engaging/logs/latent-report-%A_%a.out
#SBATCH --error=/orcd/pool/008/jrusak/diffusion-interior-engaging/logs/latent-report-%A_%a.err
set -euo pipefail
module load apptainer/1.5.2
: "${CODE_COMMIT:?Immutable evaluator required}"
: "${RUN_GROUP:?Run group required}"
root=/orcd/pool/008/jrusak/diffusion-interior-engaging
code="$root/code/$CODE_COMMIT"
image=/orcd/scratch/orcd/014/jrusak/apptainer_cache/images/diffusion-26.05-9cd36b1-amd64.sif
seed=$((1729 + ${SLURM_ARRAY_TASK_ID:-0}))
phase=observation
report=report-v1
extra=(--annual)
if [[ "${PRETRAINING_MAPS:-0}" == 1 ]]; then
  phase=om4
  report=pretraining-maps-v1
  extra=(--pretraining-maps)
fi
export SAMUDRA_CODE_COMMIT="$CODE_COMMIT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
srun --gpus-per-task=1 apptainer exec --nv --bind "$code:/pilot-code,/orcd:/orcd" "$image" \
  env PYTHONPATH=/pilot-code/src /workspace/.venv/bin/python -m samudra.experiments.diffusion_latent_report \
  --root "$root" --checkpoint "$root/runs/$RUN_GROUP/D-$seed/$phase/best.pt" \
  --output "$root/runs/$RUN_GROUP/D-$seed/$report" --members 8 "${extra[@]}"
