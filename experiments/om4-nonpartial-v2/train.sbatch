#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=om4-nonpartial-v2
#SBATCH --partition=beta
#SBATCH --qos=standard
#SBATCH --account=ny_lz1955_multiscale
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=144
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --array=0-1
set -euo pipefail
export RUN_ROOT=/projects/ny/lz1955/multiscale/jrusak/runs/om4-nonpartial-v2-20261105
export SOURCE_DATA=/projects/ny/lz1955/multiscale/jrusak/data/om4/v2026-09/om4_onedeg
export ARM=${ARM:-baseline}
if [[ ${SLURM_ARRAY_TASK_ID:-0} == 1 ]]; then export ARM=nonpartial; fi
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 WANDB_MODE=online
export WANDB_DIR="$RUN_ROOT/$ARM/wandb"
export NCCL_SOCKET_IFNAME=bond0 NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5 MELLANOX_VISIBLE_DEVICES=all
export NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export SAMUDRA_CODE_COMMIT
SAMUDRA_CODE_COMMIT=$(cat "$RUN_ROOT/commit.txt")
export SAMUDRA_CODE_REPO_URL=https://github.com/m2lines/Samudra.git
export SAMUDRA_CONTAINER_IMAGE_REF=ghcr.io/m2lines/ocean-emulator-physicsnemo@sha256:7d78e8fb9592abdf046d88e55004f15cfe34f0c8ee38b17d83e98cc05b6f82c4
on_exit() {
  rc=$?
  trap - EXIT
  if [[ ${QUALIFY_ONLY:-0} != 1 ]] && { [[ $rc != 0 ]] || [[ -f "$RUN_ROOT/baseline/workflow.complete" && -f "$RUN_ROOT/nonpartial/workflow.complete" ]]; }; then
    if [[ -f "$RUN_ROOT/callback-id" ]]; then
      python3 "$HOME/.config/codex-followup/send.py" --host beta --event-id "$(cat "$RUN_ROOT/callback-id")" || true
    fi
  fi
  exit "$rc"
}
trap on_exit EXIT
mkdir -p "$RUN_ROOT/$ARM/cache" "$WANDB_DIR" "$RUN_ROOT/output"
scontrol show job "$SLURM_JOB_ID" > "$RUN_ROOT/$ARM/allocation-${SLURM_JOB_ID}.txt"
mounts="$RUN_ROOT/source/src:/workspace/src,$RUN_ROOT/native:/workspace/native,/projects/ny/lz1955/multiscale:/projects/ny/lz1955/multiscale,$RUN_ROOT/$ARM/cache:/workspace/.data_cache"
srun --kill-on-bad-exit=1 --container-image="$SAMUDRA_CONTAINER_IMAGE_REF" --container-mount-home --container-mounts="$mounts" bash "$RUN_ROOT/source/experiments/om4-nonpartial-v2/inside.sh"
