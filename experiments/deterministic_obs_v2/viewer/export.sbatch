#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=deterministic-viewer-export
#SBATCH --account=torch_pr_347_lzanna
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx6000
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --comment=preemption=yes;requeue=true
#SBATCH --requeue
#SBATCH --chdir=/scratch/jr7309
#SBATCH --output=/scratch/jr7309/deterministic-obs-v2/viewer/export-%A_%a.out
set -euo pipefail
if ! command -v singularity >/dev/null; then
  module load singularity-ce
fi
arms=(C D)
arm=${arms[$SLURM_ARRAY_TASK_ID]}
root=/scratch/jr7309/deterministic-obs-v2/viewer
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
srun singularity exec --nv --overlay /scratch/jr7309/.apptainer-code-layers/samudra-code-7bef723cb1f60d4e202aeece6fd4c1b11eb2e539.img:ro --bind /scratch/jr7309:/scratch/jr7309 --pwd /opt/samudra-code /scratch/jr7309/.apptainer-images/ghcr.io-m2lines-ocean-emulator-physicsnemo-26.05-4f86be45a09cb9eba1b51b6273e0189941b02495.sif /usr/bin/env PYTHONPATH=/opt/samudra-code/src PYTHONUNBUFFERED=1 /workspace/.venv/bin/python "$root/export.py" --arm "$arm" --output "$root/deterministic-$arm-061126"
