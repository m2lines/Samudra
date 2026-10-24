#!/bin/bash
#SBATCH --job-name=om4-fine-train
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-12:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --array=0-1%2
set -euo pipefail
ROOT=/home/jrusak/orcd/scratch/om4-temporal-memory-20261016
PY=/home/jrusak/Ocean_Emulator/.venv/bin/python
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=2
arms=(fine_latent fine_latent_detach)
arm="${arms[$SLURM_ARRAY_TASK_ID]}"
test -f "$ROOT/FINE_QUALIFIED"
cd "$ROOT/code-fine-v1"
export WANDB_DIR="$ROOT/training/$arm"
mkdir -p "$WANDB_DIR"
"$PY" -u train.py --data "$ROOT/training-data-archive" --arm "$arm" --output "$WANDB_DIR" --max-seconds=125400 &
child=$!
trap 'kill -USR1 "$child" 2>/dev/null || true; wait "$child"' USR1 TERM
wait "$child"
if [[ -f "$WANDB_DIR/COMPLETE" ]]; then
    for selection in selected epoch70; do
        out="$ROOT/checkpoint-comparison/$arm/$selection"
        if [[ ! -f "$out/EVAL_COMPLETE" ]]; then
            "$PY" -u evaluate.py --data "$ROOT/training-data-archive" --arm "$arm" --run "$WANDB_DIR" --selection "$selection" --output "$out"
        fi
    done
fi
