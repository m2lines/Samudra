#!/bin/bash -l
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=diffusion-scratch-qualify
#SBATCH --account=mit_general
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/orcd/pool/008/jrusak/diffusion-interior-engaging/logs/scratch-qualify-%j.out
#SBATCH --error=/orcd/pool/008/jrusak/diffusion-interior-engaging/logs/scratch-qualify-%j.err
set -euo pipefail
module load apptainer/1.5.2
: "${CODE_COMMIT:?Immutable producer required}"
root=/orcd/pool/008/jrusak/diffusion-interior-engaging
code="$root/code/$CODE_COMMIT"
image=/orcd/scratch/orcd/014/jrusak/apptainer_cache/images/diffusion-26.05-9cd36b1-amd64.sif
export SAMUDRA_CODE_COMMIT="$CODE_COMMIT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
srun --gpus-per-task=1 apptainer exec --nv --bind "$code:/pilot-code,/orcd:/orcd" "$image" \
  env PYTHONPATH=/pilot-code/src /workspace/.venv/bin/python -m samudra.experiments.diffusion_observation_qualify \
  --root "$root" --output "$root/runs/scratch-qualification/$SLURM_JOB_ID-r${SLURM_RESTART_COUNT:-0}" \
  --arm B --decoder-width 192 --initialization scratch --sampling-steps 16 --fit-steps 10
