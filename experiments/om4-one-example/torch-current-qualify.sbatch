#!/bin/bash
#SBATCH --job-name=om4-bounded-range
#SBATCH --account=torch_pr_347_lzanna
#SBATCH --comment=preemption=yes;requeue=true
#SBATCH --constraint=rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-5%2
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --chdir=/scratch/jr7309
set -uo pipefail
ROOT=/scratch/jr7309/om4-current-qualify-20261020
export OMP_NUM_THREADS=4 WANDB_MODE=disabled REQUEUE_ON_USR1=1
export APPTAINER_CACHEDIR=/scratch/jr7309/apptainer-cache
export SINGULARITY_CACHEDIR=/scratch/jr7309/singularity-cache
export APPTAINER_TMPDIR="${SLURM_TMPDIR:-/tmp}"
SIF=/scratch/jr7309/.apptainer-images/ghcr.io-m2lines-ocean-emulator-physicsnemo-26.05-23587a12baeb0f39e90a69c2560a5c6a5ddbed21.sif
export SAMUDRA_SOURCE="$ROOT/main"
export OM4_COORDS="$ROOT/scales/coords.npz"
export PYTHONPATH="$ROOT/main/src"
variants=(mse spatial_100 mild25 mild10 gradient_h gradient_hz)
variant=${variants[$SLURM_ARRAY_TASK_ID]}
arm=$variant
scales="$ROOT/scales/mild25.npz"
if [[ "$variant" == mild25 || "$variant" == mild10 ]]; then
  arm=local_range
  scales="$ROOT/scales/$variant.npz"
fi
cd "$ROOT" || exit 1
sha256sum --check --quiet manifest.sha256 || exit 1
printf 'W&B mode: %s\n' "$WANDB_MODE"
apptainer exec --nv --bind /scratch:/scratch "$SIF" /workspace/.venv/bin/python -u "$ROOT/code/om4-one-example/adam_current_qualify.py" --data /scratch/jr7309/om4-temporal-memory-20261016/om4-overfit-torch-input --scales "$scales" --arm "$arm" --seed 15 --steps 1500 --output "$ROOT/results/$variant" &
child=$!
trap 'kill -USR1 "$child" 2>/dev/null || true; wait "$child"; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1
wait "$child"
rc=$?
if [[ $rc == 75 ]]; then scontrol requeue "$SLURM_JOB_ID"; exit 0; fi
exit "$rc"
