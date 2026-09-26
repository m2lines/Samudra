#!/bin/bash -l
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=diffusion-train
#SBATCH --account=mit_general
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/orcd/pool/008/jrusak/diffusion-interior-engaging/logs/train-%j.out
#SBATCH --error=/orcd/pool/008/jrusak/diffusion-interior-engaging/logs/train-%j.err
set -euo pipefail
module load apptainer/1.5.2
: "${CODE_COMMIT:?Immutable producer required}"
root=/orcd/pool/008/jrusak/diffusion-interior-engaging
code="$root/code/$CODE_COMMIT"
launcher="$root/code/${LAUNCHER_COMMIT:-$CODE_COMMIT}/scripts/diffusion_engaging_train_rank.sh"
image=/orcd/scratch/orcd/014/jrusak/apptainer_cache/images/diffusion-26.05-9cd36b1-amd64.sif
export SAMUDRA_CODE_COMMIT="$CODE_COMMIT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
srun --ntasks=4 --gpus-per-task=1 --gpu-bind=single:1 --kill-on-bad-exit=1 \
  apptainer exec --nv --bind "$code:/pilot-code,/orcd:/orcd" "$image" \
  env PYTHONPATH=/pilot-code/src bash "$launcher" "$root"
