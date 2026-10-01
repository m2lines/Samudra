#!/bin/bash -l
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=latent-endpoints
#SBATCH --account=mit_general
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/orcd/pool/008/jrusak/diffusion-interior-engaging/logs/latent-endpoints-%A_%a.out
#SBATCH --error=/orcd/pool/008/jrusak/diffusion-interior-engaging/logs/latent-endpoints-%A_%a.err
set -euo pipefail
module load apptainer/1.5.2
: "${CODE_COMMIT:?Immutable evaluator required}"
root=/orcd/pool/008/jrusak/diffusion-interior-engaging
code="$root/code/$CODE_COMMIT"
image=/orcd/scratch/orcd/014/jrusak/apptainer_cache/images/diffusion-26.05-9cd36b1-amd64.sif
seed=$((1729 + ${SLURM_ARRAY_TASK_ID:-0}))
export SAMUDRA_CODE_COMMIT="$CODE_COMMIT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
modes=(obs-adapted om4-adapted om4-pretrained)
if [[ "${NATIVE_ONLY:-0}" == 1 ]]; then modes=(om4-adapted om4-pretrained); fi
for mode in "${modes[@]}"; do
  input=om4
  checkpoint="$root/runs/latent-d192-extended-v1/D-$seed/observation/best.pt"
  extra=(--stopped-run)
  if [[ "$mode" == obs-adapted ]]; then input=observation; fi
  if [[ "$mode" == om4-pretrained ]]; then
    checkpoint="$root/runs/latent-d192-host-v1/D-$seed/om4/best.pt"
    extra=()
  fi
  srun --gpus-per-task=1 apptainer exec --nv --bind "$code:/pilot-code,/orcd:/orcd" "$image" \
    env PYTHONPATH=/pilot-code/src /workspace/.venv/bin/python -m samudra.experiments.diffusion_endpoint_maps \
    --root "$root" --checkpoint "$checkpoint" --input "$input" \
    --output "$root/reports/endpoint-members-v1/D-$seed/$mode" "${extra[@]}"
done
