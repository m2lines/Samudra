#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

#SBATCH --job-name=cm4-100yr-best-ema
#SBATCH --partition=beta
#SBATCH --qos=standard
#SBATCH --account=ny_lz1955_multiscale
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=144
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --time=02:00:00
set -euo pipefail

export TRAIN_ROOT=/projects/ny/lz1955/multiscale/jrusak/runs/cm4-onedeg-samudra-20261017
export ROOT="$TRAIN_ROOT/eval-100yr-best-ema-20261018"
export DATA_ROOT=/projects/ny/lz1955/multiscale/jrusak/data/cm4_onedeg
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export WANDB_DIR="$ROOT/wandb" MPLCONFIGDIR="$ROOT/cache/matplotlib"
export SAMUDRA_CODE_COMMIT="$(cat "$ROOT/commit.txt")"
export SAMUDRA_CODE_REPO_URL=https://github.com/m2lines/Samudra.git
export SAMUDRA_CODE_LAYER_SHA256="$(cut -d ' ' -f 1 "$ROOT/source.sha256")"
export SAMUDRA_CONTAINER_IMAGE_REF=ghcr.io/m2lines/ocean-emulator-physicsnemo@sha256:7d78e8fb9592abdf046d88e55004f15cfe34f0c8ee38b17d83e98cc05b6f82c4
on_exit() {
    rc=$?
    trap - EXIT
    python3 "$ROOT/private/send.py" --credential-file "$ROOT/private/completion.json" || true
    exit "$rc"
}
trap on_exit EXIT
mkdir -p "$ROOT/cache" "$WANDB_DIR" "$MPLCONFIGDIR"
scontrol show job "$SLURM_JOB_ID" > "$ROOT/allocation.txt"
python3 "$ROOT/private/send.py" --credential-file "$ROOT/private/completion.json" --check
mounts="$ROOT/source/src:/workspace/src,$TRAIN_ROOT/native:/workspace/native,$TRAIN_ROOT:$TRAIN_ROOT,$DATA_ROOT:$DATA_ROOT,$ROOT/cache:/workspace/.data_cache"
srun --kill-on-bad-exit=1 --container-image="$SAMUDRA_CONTAINER_IMAGE_REF" --container-mount-home --container-mounts="$mounts" bash -c '
set -euo pipefail
export PYTHONPATH=/workspace/src:/workspace/native
cd "$ROOT"
export WANDB_MODE=disabled
/workspace/.venv/bin/python "$ROOT/source/experiments/cm4-onedeg-samudra/prepare_eval_ema.py" "$TRAIN_ROOT/train.yaml" "$TRAIN_ROOT/output/cm4-onedeg-samudra-20261017/saved_nets/best_validation_ckpt.pt" "$ROOT/best-epoch40-ema.pt"
export WANDB_MODE=online
exec /workspace/.venv/bin/python "$ROOT/source/experiments/cm4-onedeg-samudra/run_eval_100yr.py" "$ROOT/eval.yaml"
'
