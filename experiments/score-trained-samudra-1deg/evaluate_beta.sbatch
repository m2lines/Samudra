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
#SBATCH --time=04:00:00

set -euo pipefail
export CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-/projects/ny/lz1955/multiscale/jrusak/runs/score-samudra-1deg-20261011}"
: "${SOURCE_ROOT:?Set SOURCE_ROOT to an immutable source snapshot}"
arm="$1"
mode="$2"
case "$arm" in A|B|C|D) ;; *) exit 2 ;; esac
case "$mode" in qualification|full) ;; *) exit 2 ;; esac
: "${FOLLOWUP_CREDENTIAL:?Set FOLLOWUP_CREDENTIAL to the private callback file}"
on_exit() {
    rc=$?
    trap - EXIT
    python3 "$CAMPAIGN_ROOT/private/send.py" --credential-file "$FOLLOWUP_CREDENTIAL" || true
    exit "$rc"
}
trap on_exit EXIT

export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 WANDB_MODE=online
export NCCL_SOCKET_IFNAME=bond0 NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5 MELLANOX_VISIBLE_DEVICES=all
export NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export SAMUDRA_CODE_COMMIT="$(cat "$SOURCE_ROOT/COMMIT")"
export SAMUDRA_CONTAINER_IMAGE_REF=ghcr.io/m2lines/ocean-emulator-physicsnemo@sha256:7d78e8fb9592abdf046d88e55004f15cfe34f0c8ee38b17d83e98cc05b6f82c4
export DATA_ROOT=/projects/ny/lz1955/multiscale/jrusak/data/om4/v2026-09/om4_onedeg
export EVAL_OUTPUT="$CAMPAIGN_ROOT/outputs/eval-$mode-$arm-$SLURM_JOB_ID"
export WANDB_DIR="$EVAL_OUTPUT"
config_arm="$arm"
checkpoint="$CAMPAIGN_ROOT/outputs/train-$arm/saved_nets/ema_ckpt.pt"
if [[ "$arm" == A ]]; then
    config_arm=B
    checkpoint=/projects/ny/lz1955/multiscale/jrusak/runs/current-stochastic-depth-1deg-20261006/control/output/current-1deg-control-ga2-210664/saved_nets/ema_ckpt.pt
fi
mkdir -p "$EVAL_OUTPUT" "$CAMPAIGN_ROOT/cache/eval-$arm-$SLURM_JOB_ID" "$CAMPAIGN_ROOT/allocations"
scontrol show job "$SLURM_JOB_ID" > "$CAMPAIGN_ROOT/allocations/$SLURM_JOB_ID.txt"
# Connectivity checks are separate from the scientific workload's startup.
mounts="$SOURCE_ROOT/src:/workspace/src,$CAMPAIGN_ROOT/native:/workspace/native,$CAMPAIGN_ROOT:$CAMPAIGN_ROOT,$DATA_ROOT:$DATA_ROOT,$CAMPAIGN_ROOT/cache/eval-$arm-$SLURM_JOB_ID:/workspace/.data_cache"
if [[ "$arm" == A ]]; then
    checkpoint_dir="$(dirname "$checkpoint")"
    mounts+=",$checkpoint_dir:$checkpoint_dir:ro"
fi
srun --kill-on-bad-exit=1 --container-image="$SAMUDRA_CONTAINER_IMAGE_REF" --container-mount-home --container-mounts="$mounts" bash -c '
set -euo pipefail
export PYTHONPATH=/workspace/src:/workspace/native
cd "$SOURCE_ROOT/experiments/score-trained-samudra-1deg"
args=()
if [[ "$2" == qualification ]]; then args+=(--origin-limit 4); fi
exec /workspace/.venv/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 evaluate_score.py "$CAMPAIGN_ROOT/configs/train-$3.yaml" "$4" "$EVAL_OUTPUT" --arm "$1" --members 16 "${args[@]}"
' bash "$arm" "$mode" "$config_arm" "$checkpoint"
