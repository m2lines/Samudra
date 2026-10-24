#!/bin/bash
#SBATCH --job-name=om4-coastal-confirm
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-3%2
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
set -uo pipefail
ROOT=/orcd/scratch/orcd/014/jrusak/om4-one-example-20261019
test -f "$ROOT/INPUTS_VERIFIED" || exit 1
export OMP_NUM_THREADS=4 WANDB_MODE=disabled REQUEUE_ON_USR1=1
export APPTAINER_CACHEDIR="$ROOT/cache"
export APPTAINER_TMPDIR="${SLURM_TMPDIR:-/tmp}"
APPTAINER=/orcd/software/core/001/pkg/apptainer/1.5.2/bin/apptainer
SIF="$ROOT/images/ghcr.io-m2lines-ocean-emulator-physicsnemo-26.05-23587a12baeb0f39e90a69c2560a5c6a5ddbed21.sif"
contexts=(seed16_oct seed15_jan)
boundaries=(interior wet_neighbors)
if [[ ${MODE:-train} == pilot ]]; then
    context=${contexts[$SLURM_ARRAY_TASK_ID]}
    boundary=${boundaries[$SLURM_ARRAY_TASK_ID]}
    steps=2
    output="$ROOT/pilot/$context/$boundary"
else
    test -f "$ROOT/QUALIFIED" || exit 1
    context=${contexts[$((SLURM_ARRAY_TASK_ID / 2))]}
    boundary=${boundaries[$((SLURM_ARRAY_TASK_ID % 2))]}
    steps=1000
    output="$ROOT/coastal-confirm/$context/$boundary"
fi
data="$ROOT/data/oct"
if [[ $context == seed15_jan ]]; then data="$ROOT/data/jan"; fi
"$APPTAINER" exec --nv --bind /orcd:/orcd "$SIF" /workspace/.venv/bin/python -u \
    "$ROOT/code/om4-one-example/lbfgs_coastal_extended.py" \
    --data "$data" --source "$ROOT/source/$context" --weight 100 \
    --boundary "$boundary" --total-outer-steps "$steps" --output "$output" &
child=$!
trap 'kill -USR1 "$child" 2>/dev/null || true; wait "$child"; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1
wait "$child"
rc=$?
if [[ $rc == 75 ]]; then scontrol requeue "$SLURM_JOB_ID"; exit 0; fi
exit "$rc"
