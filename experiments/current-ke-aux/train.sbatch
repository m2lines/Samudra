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
#SBATCH --time=12:00:00
set -euo pipefail
mode="$1"
case "$mode" in aligned|seasonal) ;; *) exit 2;; esac
export KE_ROOT=/projects/ny/lz1955/multiscale/jrusak/runs/current-ke-aux-20261010
export KE_DATA=/projects/ny/lz1955/multiscale/jrusak/data/om4/v2026-09
export KE_MODE="$mode"
export ROOT="$KE_ROOT/$mode"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 WANDB_MODE=online WANDB_DIR="$ROOT/wandb"
export NCCL_SOCKET_IFNAME=bond0 NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5 MELLANOX_VISIBLE_DEVICES=all
export NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export SAMUDRA_CODE_COMMIT=$(cat "$KE_ROOT/commit.txt")
export SAMUDRA_CODE_REPO_URL=https://github.com/m2lines/Samudra.git
export SAMUDRA_CONTAINER_IMAGE_REF=ghcr.io/m2lines/ocean-emulator-physicsnemo@sha256:7d78e8fb9592abdf046d88e55004f15cfe34f0c8ee38b17d83e98cc05b6f82c4
mkdir -p "$ROOT/cache" "$ROOT/output" "$WANDB_DIR"
scontrol show job "$SLURM_JOB_ID" > "$ROOT/allocation.txt"
srun --kill-on-bad-exit=1 --container-image="$SAMUDRA_CONTAINER_IMAGE_REF" --container-mount-home --container-mounts="$KE_ROOT/source/src:/workspace/src,$KE_ROOT/native:/workspace/native,$KE_ROOT:$KE_ROOT,$KE_DATA:$KE_DATA,$ROOT/cache:/workspace/.data_cache" bash -c '
set -euo pipefail
export PYTHONPATH=/workspace/src:/workspace/native
cd "$ROOT"
# Exercise real four-rank forward/backward, EMA, dynamic weights, and rollout validation.
WANDB_MODE=disabled /workspace/.venv/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 -m samudra.train "$KE_ROOT/$KE_MODE-train.yaml" --backend=nccl --debug=true --epochs=2 --experiment.wandb.mode=disabled --experiment.name="ke-${KE_MODE}-preflight-${SLURM_JOB_ID}" --experiment.base_output_dir="$ROOT/preflight"
# Fresh process and initialization; preflight weights never enter production.
exec /workspace/.venv/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 -m samudra.train "$KE_ROOT/$KE_MODE-train.yaml" --backend=nccl --experiment.name="current-ke-${KE_MODE}-seed15-${SLURM_JOB_ID}" --experiment.base_output_dir="$ROOT/output"
'
