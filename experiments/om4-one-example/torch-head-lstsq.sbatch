#!/bin/bash
#SBATCH --job-name=om4-one-head-ls
#SBATCH --account=torch_pr_347_lzanna
#SBATCH --comment=preemption=yes;requeue=true
#SBATCH --constraint=rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --chdir=/scratch/jr7309
set -uo pipefail
ROOT=/scratch/jr7309/om4-one-example-20261018
export OMP_NUM_THREADS=4 WANDB_MODE=disabled REQUEUE_ON_USR1=1
export APPTAINER_CACHEDIR=/scratch/jr7309/apptainer-cache
export SINGULARITY_CACHEDIR=/scratch/jr7309/singularity-cache
export APPTAINER_TMPDIR="${SLURM_TMPDIR:-/tmp}"
SIF=/scratch/jr7309/.apptainer-images/ghcr.io-m2lines-ocean-emulator-physicsnemo-26.05-23587a12baeb0f39e90a69c2560a5c6a5ddbed21.sif
apptainer exec --nv --bind /scratch:/scratch "$SIF" /workspace/.venv/bin/python -u "$ROOT/code/om4-one-example/head_lstsq.py" --data /scratch/jr7309/om4-temporal-memory-20261016/om4-overfit-torch-input --checkpoint /scratch/jr7309/om4-temporal-memory-20261016/overfit-one/latest.pt --output "$ROOT/head-lstsq" &
child=$!
trap 'kill -TERM "$child" 2>/dev/null || true; wait "$child"; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1
wait "$child"
