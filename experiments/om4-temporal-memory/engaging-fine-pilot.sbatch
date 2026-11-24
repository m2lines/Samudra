#!/bin/bash
#SBATCH --job-name=om4-fine-pilot
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:20:00
set -euo pipefail
ROOT=/home/jrusak/orcd/scratch/om4-temporal-memory-20261016
PY=/home/jrusak/Ocean_Emulator/.venv/bin/python
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=2
cd "$ROOT/code-fine-v1"
"$PY" -m pytest -q test_mechanisms.py
for arm in fine_latent fine_latent_detach; do
    export WANDB_DIR="$ROOT/fine-pilot/$arm"
    mkdir -p "$WANDB_DIR"
    "$PY" -u train.py --data "$ROOT/training-data-archive" --arm "$arm" --output "$WANDB_DIR" --smoke
    test -f "$WANDB_DIR/latest.pt"
done
"$PY" qualify_fine.py --root "$ROOT"
