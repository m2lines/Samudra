#!/bin/bash
#SBATCH --job-name=om4-structure-resume
#SBATCH --partition=mit_preemptable
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-2%3
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
set -euo pipefail
: "${ROOT:?}" "${PHASE:?}"
cd "$ROOT"
test -f TRANSFER_VERIFIED
test -f DATA_AUDIT_VERIFIED
if [[ "$PHASE" == full ]]; then test -f ENGAGING_QUALIFIED; fi
export PYTHONPATH="$ROOT/code/src"
export WANDB_MODE=online OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2
export APPTAINER_CACHEDIR="$ROOT/apptainer-cache"
export APPTAINER_TMPDIR="${SLURM_TMPDIR:-/tmp}"
export WANDB_DIR="$ROOT" WANDB_CACHE_DIR="$ROOT/wandb-cache" DATA_CACHE_DIR="$ROOT/data-cache"
SIF=/orcd/scratch/orcd/014/jrusak/om4-one-example-20261019/images/ghcr.io-m2lines-ocean-emulator-physicsnemo-26.05-23587a12baeb0f39e90a69c2560a5c6a5ddbed21.sif
APPTAINER=/orcd/software/core/001/pkg/apptainer/1.5.2/bin/apptainer
arms=(local spatial gradient)
arm=${arms[$SLURM_ARRAY_TASK_ID]}
"$APPTAINER" exec --nv --bind /orcd:/orcd "$SIF" /workspace/.venv/bin/python -u -m samudra.train "$ROOT/configs/engaging-$PHASE-$arm.yaml" &
child=$!
trap 'scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1
wait "$child"
