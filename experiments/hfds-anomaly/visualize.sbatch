#!/bin/bash
#SBATCH --job-name=hfds-viz
#SBATCH --partition=mit_preemptable
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
set -euo pipefail
ROOT=/orcd/pool/008/jrusak/om4-hfds-ablation-20261021
cd "$ROOT"
export PYTHONPATH="$ROOT/code/src" PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export APPTAINERENV_PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2
export WANDB_DIR="$ROOT" WANDB_CACHE_DIR="$ROOT/cache/wandb" DATA_CACHE_DIR="$ROOT/cache/data"
export APPTAINER_CACHEDIR="$ROOT/cache/apptainer" APPTAINER_TMPDIR="${SLURM_TMPDIR:-/tmp}"
SIF=/orcd/scratch/orcd/014/jrusak/om4-one-example-20261019/images/ghcr.io-m2lines-ocean-emulator-physicsnemo-26.05-23587a12baeb0f39e90a69c2560a5c6a5ddbed21.sif
APPTAINER=/orcd/software/core/001/pkg/apptainer/1.5.2/bin/apptainer
export MPLBACKEND=Agg APPTAINERENV_MPLBACKEND=Agg
run() { "$APPTAINER" exec --nv --bind /orcd:/orcd --bind "$ROOT/runtime-fixes/plotting.py:/workspace/.venv/lib/python3.12/site-packages/xarrayutils/plotting.py:ro" "$SIF" /workspace/.venv/bin/python -u "$@"; }
labels=(selected final-ema)
label=${labels[$SLURM_ARRAY_TASK_ID]}
run "$ROOT/visualize.py" "$ROOT/configs/viz-$label.yaml" &
child=$!
trap 'scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1
wait "$child"
