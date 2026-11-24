#!/bin/bash
#SBATCH --job-name=om4-standard-suite
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --array=0-7%2
set -euo pipefail
ROOT=/home/jrusak/orcd/scratch/om4-temporal-memory-20261016
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 MPLCONFIGDIR="$ROOT/matplotlib"
export PYTHONPATH="$ROOT/code-suite-v1/repository-src"
export PYTHONUNBUFFERED=1
arms=(full detach latent_full latent_only)
selections=(selected epoch70)
arm=${arms[$((SLURM_ARRAY_TASK_ID / 2))]}
selection=${selections[$((SLURM_ARRAY_TASK_ID % 2))]}
cd "$ROOT/code-suite-v1"
/home/jrusak/Ocean_Emulator/.venv/bin/python -u run_standard_suite.py --root "$ROOT" --rollout "$ROOT/standard-suite/rollouts/$arm/$selection" --observations "$ROOT/standard-suite/observations" --output "$ROOT/standard-suite/results/$arm/$selection"
