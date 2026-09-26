#!/bin/bash -l
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=diffusion-qualify
#SBATCH --account=mit_general
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=/orcd/pool/008/jrusak/diffusion-interior-engaging/logs/qualify-%j.out
#SBATCH --error=/orcd/pool/008/jrusak/diffusion-interior-engaging/logs/qualify-%j.err
set -euo pipefail
module load apptainer/1.5.2
: "${CODE_COMMIT:?An immutable producer is required}"
root=/orcd/pool/008/jrusak/diffusion-interior-engaging
code="$root/code/$CODE_COMMIT"
image=/orcd/scratch/orcd/014/jrusak/apptainer_cache/images/diffusion-26.05-9cd36b1-amd64.sif
test -f "$root/DATA_READY.json"
test -f "$root/checkpoints/selection.json"
test -f "$image"
export SAMUDRA_CODE_COMMIT="$CODE_COMMIT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
srun --ntasks=4 --gpus-per-task=1 --gpu-bind=single:1 --kill-on-bad-exit=1 \
  apptainer exec --nv --bind "$code:/pilot-code,/orcd:/orcd" "$image" \
  env PYTHONPATH=/pilot-code/src bash /pilot-code/scripts/diffusion_beta_qualify_rank.sh "$root"
