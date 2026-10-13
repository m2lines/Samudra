#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

#SBATCH --partition=beta
#SBATCH --qos=standard
#SBATCH --account=ny_lz1955_multiscale
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=144
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=24:00:00
set -euo pipefail
export PAIR=/projects/ny/lz1955/multiscale/jrusak/runs/current-stochastic-depth-1deg-20261006
variant="$1"
case "$variant" in control|sd01) ;; *) exit 2 ;; esac
export ROOT="$PAIR/$variant"
export DATA_ROOT=/mnt/home/jrusak/data/om4/v2026-09/om4_onedeg
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 WANDB_MODE=online
export WANDB_DIR="$ROOT/wandb"
export NCCL_SOCKET_IFNAME=bond0 NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5 MELLANOX_VISIBLE_DEVICES=all
export NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export SAMUDRA_CODE_COMMIT="$(cat "$PAIR/commit.txt")"
export SAMUDRA_CODE_REPO_URL=https://github.com/m2lines/Samudra.git
export SAMUDRA_CONTAINER_IMAGE_REF=ghcr.io/m2lines/ocean-emulator-physicsnemo@sha256:7d78e8fb9592abdf046d88e55004f15cfe34f0c8ee38b17d83e98cc05b6f82c4
mkdir -p "$ROOT/cache" "$WANDB_DIR" "$ROOT/output"
scontrol show job "$SLURM_JOB_ID" > "$ROOT/allocation.txt"
mounts="$PAIR/source/src:/workspace/src,$PAIR/native:/workspace/native,$PAIR:$PAIR,$DATA_ROOT:$DATA_ROOT,$ROOT/cache:/workspace/.data_cache"
srun --kill-on-bad-exit=1 --container-image="$SAMUDRA_CONTAINER_IMAGE_REF" --container-mount-home --container-mounts="$mounts" bash -c '
set -euo pipefail
export PYTHONPATH=/workspace/src:/workspace/native
cd "$ROOT"
/workspace/.venv/bin/python -c "import torch,samudra,samudra_rust_loader; print(torch.__version__,samudra.__file__,samudra_rust_loader.__file__)"
exec /workspace/.venv/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 -m samudra.train "$ROOT/train.yaml" --backend=nccl --experiment.name="current-1deg-${1}-ga2-${SLURM_JOB_ID}" --experiment.data_root="$DATA_ROOT" --experiment.base_output_dir="$ROOT/output"
' bash "$variant"
