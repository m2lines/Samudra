#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

#SBATCH --job-name=cm4-onedeg-samudra
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
#SBATCH --time=24:00:00
set -euo pipefail

export ROOT=/projects/ny/lz1955/multiscale/jrusak/runs/cm4-onedeg-samudra-20261017
export DATA_ROOT=/projects/ny/lz1955/multiscale/jrusak/data/cm4_onedeg
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export WANDB_DIR="$ROOT/wandb"
export MPLCONFIGDIR="$ROOT/cache/matplotlib"
export NCCL_SOCKET_IFNAME=bond0 NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5 MELLANOX_VISIBLE_DEVICES=all
export NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export SAMUDRA_CODE_COMMIT="$(cat "$ROOT/commit.txt")"
export SAMUDRA_CODE_REPO_URL=https://github.com/m2lines/Samudra.git
export SAMUDRA_CODE_LAYER_SHA256="$(cut -d ' ' -f 1 "$ROOT/source.sha256")"
export SAMUDRA_CONTAINER_IMAGE_REF=ghcr.io/m2lines/ocean-emulator-physicsnemo@sha256:7d78e8fb9592abdf046d88e55004f15cfe34f0c8ee38b17d83e98cc05b6f82c4
mode="${1:-train}"
case "$mode" in train|qualify) ;; *) exit 2 ;; esac
if [[ "$mode" == train ]]; then
    on_exit() {
        rc=$?
        trap - EXIT
        python3 "$ROOT/private/send.py" --credential-file "$ROOT/private/completion.json" || true
        exit "$rc"
    }
    trap on_exit EXIT
fi
mkdir -p "$ROOT/cache" "$WANDB_DIR" "$MPLCONFIGDIR"
scontrol show job "$SLURM_JOB_ID" > "$ROOT/allocation-$mode.txt"
if [[ "$mode" == train ]]; then
    python3 "$ROOT/private/send.py" --credential-file "$ROOT/private/completion.json" --check
    # Refuse full training if real-data qualification did not succeed.
    python3 -c 'import json,os; assert json.load(open(os.environ["ROOT"]+"/qualification.json"))["passed"]'
fi
mounts="$ROOT/source/src:/workspace/src,$ROOT/native:/workspace/native,$ROOT:$ROOT,$DATA_ROOT:$DATA_ROOT,$ROOT/cache:/workspace/.data_cache"
export RUN_MODE="$mode"
srun --kill-on-bad-exit=1 --container-image="$SAMUDRA_CONTAINER_IMAGE_REF" --container-mount-home --container-mounts="$mounts" bash -c '
set -euo pipefail
export PYTHONPATH=/workspace/src:/workspace/native
cd "$ROOT"
/workspace/.venv/bin/python -c "import torch,samudra,samudra_rust_loader; print(torch.__version__,samudra.__file__,samudra_rust_loader.__file__)"
if [[ "$RUN_MODE" == qualify ]]; then
    export WANDB_MODE=disabled
    exec /workspace/.venv/bin/python "$ROOT/source/experiments/cm4-onedeg-samudra/qualify.py" "$ROOT/train.yaml" "$ROOT/qualification.json"
fi
export WANDB_MODE=online
exec /workspace/.venv/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 -m samudra.train "$ROOT/train.yaml" --backend=nccl
'
