#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=obs-D-report
#SBATCH --account=torch_pr_347_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --chdir=/scratch/jr7309
#SBATCH --output=/scratch/jr7309/slurm-%j.out
#SBATCH --error=/scratch/jr7309/slurm-%j.err
set -euo pipefail
arm="${1:?Pass primary, adapter-only, or scratch}"
case "$arm" in primary|adapter-only|scratch) ;; *) exit 2 ;; esac
root=/scratch/jr7309/runs/2026-09-22-observation-D
data=/scratch/jr7309/data/obs-d-pilot
layer=/scratch/jr7309/.apptainer-code-layers/samudra-code-d8949bb15d9fa92e6c8c94702092360bbe9005ef.img
sif=/scratch/jr7309/.apptainer-images/ghcr.io-m2lines-ocean-emulator-physicsnemo-26.05-9cd36b1bdcf4921fb027494afa4157e217818ccd.sif
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
srun /share/apps/apptainer/bin/apptainer exec --overlay "$layer:ro" \
  --bind "$root:$root,$data:$data" --env PYTHONPATH=/opt/samudra-code/src "$sif" \
  /workspace/.venv/bin/python "$root/analyze_observation_predictions.py" \
  --evaluation "$root/$arm-evaluation" --grid "$data/grid.npz" \
  --selection-reference "$root/fitting/selection-reference.json" \
  --output "$root/$arm-evaluation/anomaly-year-report.json"
