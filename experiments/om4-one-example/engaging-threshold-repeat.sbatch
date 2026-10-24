#!/bin/bash
#SBATCH --job-name=om4-threshold-repeat
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --array=0-1%2
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
set -uo pipefail
ROOT=/orcd/scratch/orcd/014/jrusak/om4-one-example-20261019
test -f "$ROOT/JANUARY_PAIR_VERIFIED" || exit 1
export OMP_NUM_THREADS=4 WANDB_MODE=disabled REQUEUE_ON_USR1=1
export APPTAINER_CACHEDIR="$ROOT/cache"
export APPTAINER_TMPDIR="${SLURM_TMPDIR:-/tmp}"
APPTAINER=/orcd/software/core/001/pkg/apptainer/1.5.2/bin/apptainer
SIF="$ROOT/images/ghcr.io-m2lines-ocean-emulator-physicsnemo-26.05-23587a12baeb0f39e90a69c2560a5c6a5ddbed21.sif"
names=(default lowered)
thresholds=(1e-10 1e-20)
output="$ROOT/curvature-threshold-jan/${names[$SLURM_ARRAY_TASK_ID]}"
test -f "$output/RESUME_COPY_VERIFIED" || exit 1
"$APPTAINER" exec --nv --bind /orcd:/orcd "$SIF" /workspace/.venv/bin/python -u \
    "$ROOT/code/om4-one-example/lbfgs_threshold.py" \
    --data "$ROOT/data/jan" --source "$ROOT/source/seed15_jan" --weight 100 \
    --boundary interior --curvature-threshold "${thresholds[$SLURM_ARRAY_TASK_ID]}" \
    --total-outer-steps 2500 --output "$output" &
child=$!
trap 'kill -USR1 "$child" 2>/dev/null || true; wait "$child"; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1
wait "$child"
rc=$?
if [[ $rc == 75 ]]; then scontrol requeue "$SLURM_JOB_ID"; exit 0; fi
exit "$rc"
