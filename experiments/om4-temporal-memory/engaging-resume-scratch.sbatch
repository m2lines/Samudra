#!/bin/bash
#SBATCH --job-name=om4-memory-train
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-12:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --array=0-3%4
set -euo pipefail
ROOT=/home/jrusak/orcd/scratch/om4-temporal-memory-20261016
PY=/home/jrusak/Ocean_Emulator/.venv/bin/python
export OMP_NUM_THREADS=8
arms=(full detach latent_full latent_only)
arm="${arms[$SLURM_ARRAY_TASK_ID]}"
test -f "$HOME/orcd/scratch/om4-migration-20261018/COPIES_VERIFIED"
test -f "$ROOT/QUALIFIED"
test -f "$ROOT/training-data-archive/COMPLETE"
cd "$ROOT/code-v4"
export WANDB_DIR="$ROOT/training/$arm"
export MPLCONFIGDIR="$ROOT/matplotlib"
test -f "$ROOT/training/$arm/latest.pt"
# Python checkpoints at the next effective-batch boundary on signal or deadline.
"$PY" -u train.py --data "$ROOT/training-data-archive" --arm "$arm" \
  --output "$ROOT/training/$arm" --max-seconds=125400 &
child=$!
trap 'kill -USR1 "$child" 2>/dev/null || true; wait "$child"' USR1 TERM
wait "$child"
if [[ -f "$ROOT/training/$arm/COMPLETE" ]]; then
  "$PY" -u evaluate.py --data "$ROOT/training-data-archive" --arm "$arm" \
    --run "$ROOT/training/$arm"
fi
