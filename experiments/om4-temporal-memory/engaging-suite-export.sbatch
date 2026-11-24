#!/bin/bash
#SBATCH --job-name=om4-suite-export
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-7%2
set -euo pipefail
ROOT=/home/jrusak/orcd/scratch/om4-temporal-memory-20261016
PY=/home/jrusak/Ocean_Emulator/.venv/bin/python
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=2
arms=(full detach latent_full latent_only)
selections=(selected epoch70)
arm=${arms[$((SLURM_ARRAY_TASK_ID / 2))]}
selection=${selections[$((SLURM_ARRAY_TASK_ID % 2))]}
cd "$ROOT/code-suite-v1"
"$PY" -u export_suite_rollout.py --root "$ROOT" --arm "$arm" --selection "$selection" --origin=0 --output "$ROOT/standard-suite/rollouts/$arm/$selection"
