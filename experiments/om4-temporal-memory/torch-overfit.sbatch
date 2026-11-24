#!/bin/bash
#SBATCH --job-name=om4-overfit-one
#SBATCH --account=torch_pr_347_lzanna
#SBATCH --partition=rtx6000_lzanna
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --chdir=/scratch/jr7309
set -euo pipefail
if ! command -v apptainer >/dev/null; then
  module load apptainer || module load singularity-ce
fi
ROOT=/scratch/jr7309/om4-temporal-memory-20261016
export OMP_NUM_THREADS=8
export APPTAINER_CACHEDIR=/scratch/jr7309/apptainer-cache
export SINGULARITY_CACHEDIR=/scratch/jr7309/singularity-cache
export APPTAINER_TMPDIR="${SLURM_TMPDIR:-/tmp}"
export WANDB_MODE=disabled
SIF=/scratch/jr7309/.apptainer-images/ghcr.io-m2lines-ocean-emulator-physicsnemo-26.05-23587a12baeb0f39e90a69c2560a5c6a5ddbed21.sif
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 30 > "$ROOT/gpu-${SLURM_JOB_ID}.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
apptainer exec --nv --bind /scratch:/scratch "$SIF" /workspace/.venv/bin/python -u \
 "$ROOT/code/overfit_one.py" --data "$ROOT/om4-overfit-torch-input" --output "$ROOT/overfit-one" --steps 5000
