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
#SBATCH --time=01:00:00

set -euo pipefail
export CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-/projects/ny/lz1955/multiscale/jrusak/runs/score-samudra-1deg-20261011}"
: "${SOURCE_ROOT:?Set SOURCE_ROOT to an immutable source snapshot}"
: "${FOLLOWUP_CREDENTIAL:?Set FOLLOWUP_CREDENTIAL to the private callback file}"
mode="$1"
case "$mode" in qualification|full|export) ;; *) exit 2 ;; esac
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
export EVAL_OUTPUT="$CAMPAIGN_ROOT/outputs/continuous-$mode-$SLURM_JOB_ID"
export WANDB_DIR="$EVAL_OUTPUT"
control_root=/projects/ny/lz1955/multiscale/jrusak/runs/current-stochastic-depth-1deg-20261006/control/output/current-1deg-control-ga2-210664/saved_nets
mkdir -p "$EVAL_OUTPUT" "$CAMPAIGN_ROOT/cache/continuous-$SLURM_JOB_ID" "$CAMPAIGN_ROOT/allocations"
scontrol show job "$SLURM_JOB_ID" > "$CAMPAIGN_ROOT/allocations/$SLURM_JOB_ID.txt"
python3 "$CAMPAIGN_ROOT/private/send.py" --credential-file "$FOLLOWUP_CREDENTIAL" --check || true
mounts="$SOURCE_ROOT/src:/workspace/src,$CAMPAIGN_ROOT/native:/workspace/native,$CAMPAIGN_ROOT:$CAMPAIGN_ROOT,$DATA_ROOT:$DATA_ROOT,$control_root:$control_root:ro,$CAMPAIGN_ROOT/cache/continuous-$SLURM_JOB_ID:/workspace/.data_cache"
srun --input=none --kill-on-bad-exit=1 --container-image="$SAMUDRA_CONTAINER_IMAGE_REF" --container-mount-home --container-mounts="$mounts" bash -c '
set -euo pipefail
export PYTHONPATH=/workspace/src:/workspace/native
cd "$SOURCE_ROOT/experiments/score-trained-samudra-1deg"
args=()
if [[ "$1" == qualification ]]; then args+=(--max-steps 3); fi
if [[ "$1" == export ]]; then args+=(--write-rollouts --verify-replay "$CAMPAIGN_ROOT/outputs/continuous-full-212420"); fi
exec /workspace/.venv/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 diagnose_score.py "$CAMPAIGN_ROOT" "$EVAL_OUTPUT" "${args[@]}"
' bash "$mode"
