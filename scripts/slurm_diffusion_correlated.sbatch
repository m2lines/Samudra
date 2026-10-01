#!/bin/bash -l
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=diffusion-correlated
#SBATCH --account=mit_general
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/orcd/pool/008/jrusak/diffusion-interior-engaging/logs/correlated-%A_%a.out
#SBATCH --error=/orcd/pool/008/jrusak/diffusion-interior-engaging/logs/correlated-%A_%a.err
set -euo pipefail
module load apptainer/1.5.2
: "${CODE_COMMIT:?Immutable producer required}"
root=/orcd/pool/008/jrusak/diffusion-interior-engaging
code="$root/code/$CODE_COMMIT"
image=/orcd/scratch/orcd/014/jrusak/apptainer_cache/images/diffusion-26.05-9cd36b1-amd64.sif
arm=white
correlation=0.0
if [[ "${SLURM_ARRAY_TASK_ID:-0}" == 1 ]]; then arm=correlated; correlation=0.5; fi
export SAMUDRA_CODE_COMMIT="$CODE_COMMIT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
srun --gpus-per-task=1 apptainer exec --nv --bind "$code:/pilot-code,/orcd:/orcd" "$image" \
 env PYTHONPATH=/pilot-code/src /workspace/.venv/bin/python -m samudra.experiments.diffusion_correlated_pilot \
 --root "$root" --checkpoint "$root/runs/latent-d192-v2/D-1729/om4/best.pt" \
 --output "$root/runs/correlated-noise-scratch-v2/$arm" --updates 2000 --hours 1.5 --correlation "$correlation"
