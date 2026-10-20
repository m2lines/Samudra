#!/bin/bash
#SBATCH --job-name=om4-suite-obs
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
set -euo pipefail
ROOT=/home/jrusak/orcd/scratch/om4-temporal-memory-20261016
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2
cd "$ROOT/code-suite-v1"
/home/jrusak/Ocean_Emulator/.venv/bin/python -u stage_suite_observations.py --output "$ROOT/standard-suite/observations"
