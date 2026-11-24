#!/bin/bash
#SBATCH --job-name=om4-ema-maps
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --array=0-3%4
set -euo pipefail
ROOT=/home/jrusak/orcd/scratch/om4-temporal-memory-20261016
PY=/home/jrusak/Ocean_Emulator/.venv/bin/python
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=2
arms=(full detach latent_full latent_only)
arm="${arms[$SLURM_ARRAY_TASK_ID]}"
cd "$ROOT/code-fine-v1"
for selection in selected epoch70; do
    out="$ROOT/checkpoint-comparison/$arm/$selection"
    if [[ ! -f "$out/EVAL_COMPLETE" ]]; then
        "$PY" -u evaluate.py --data "$ROOT/training-data-archive" --arm "$arm" --run "$ROOT/training/$arm" --selection "$selection" --output "$out"
    fi
done
