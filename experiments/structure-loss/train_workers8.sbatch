#!/bin/bash
#SBATCH --job-name=om4-structure
#SBATCH --account=torch_pr_347_lzanna
#SBATCH --comment=preemption=yes;requeue=true
#SBATCH --constraint=rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --array=0-2%3
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
set -uo pipefail
: "${ROOT:?}" "${SIF:?}" "${MODE:?}"
cd "$ROOT" || exit 1
sha256sum --check --quiet manifest.sha256 || exit 1
test -f "$ROOT/PREPARED" || exit 1
if [[ "$MODE" == full ]]; then test -f "$ROOT/PILOTS_VERIFIED" || exit 1; fi
arms=(local spatial gradient)
arm=${arms[$SLURM_ARRAY_TASK_ID]}
export PYTHONPATH="$ROOT/code/src"
export WANDB_MODE=online OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2
export APPTAINER_TMPDIR="${SLURM_TMPDIR:-/tmp}"
export APPTAINER_CACHEDIR="/scratch/$USER/apptainer-cache"
export WANDB_DIR="$ROOT"
export WANDB_CACHE_DIR="$ROOT/wandb-cache"
export DATA_CACHE_DIR="$ROOT/data-cache"
apptainer exec --nv --bind /scratch:/scratch "$SIF" /workspace/.venv/bin/python -u -m samudra.train "$ROOT/configs/$MODE-$arm.yaml" --data.loading.num_workers=8 &
child=$!
trap 'scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1
wait "$child"
exit $?
