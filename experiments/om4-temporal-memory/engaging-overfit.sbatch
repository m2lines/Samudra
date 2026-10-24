#!/bin/bash
#SBATCH --job-name=om4-overfit-one
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
set -euo pipefail
ROOT=/orcd/data/abodner/002/jrusak/om4-temporal-memory-20261016
export OMP_NUM_THREADS=8
cd "$ROOT/code-overfit-v1"
/home/jrusak/Ocean_Emulator/.venv/bin/python -u overfit_one.py \
 --data "$ROOT/diagnostic-data-archive" --output "$ROOT/overfit-one" --steps 5000
