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
#SBATCH --time=18:00:00

set -euo pipefail
export CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-/projects/ny/lz1955/multiscale/jrusak/runs/score-samudra-1deg-20261011}"
arm="$1"
mode="$2"
case "$arm" in B|C|D) ;; *) exit 2 ;; esac
case "$mode" in qualification|train) ;; *) exit 2 ;; esac
export DATA_ROOT=/projects/ny/lz1955/multiscale/jrusak/data/om4/v2026-09/om4_onedeg
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 WANDB_MODE=online
export WANDB_DIR="$CAMPAIGN_ROOT/wandb/$mode-$arm"
export NCCL_SOCKET_IFNAME=bond0 NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5 MELLANOX_VISIBLE_DEVICES=all
export NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export SAMUDRA_CODE_COMMIT="$(cat "$CAMPAIGN_ROOT/source-commit.txt")"
export SAMUDRA_CODE_REPO_URL=https://github.com/m2lines/Samudra.git
export SAMUDRA_CONTAINER_IMAGE_REF=ghcr.io/m2lines/ocean-emulator-physicsnemo@sha256:7d78e8fb9592abdf046d88e55004f15cfe34f0c8ee38b17d83e98cc05b6f82c4
callback="$CAMPAIGN_ROOT/private/$SLURM_JOB_ID.json"
if [[ -f "$callback" ]]; then
    python3 "$CAMPAIGN_ROOT/private/send.py" --credential-file "$callback" --check
    on_exit() {
        rc=$?
        trap - EXIT
        python3 "$CAMPAIGN_ROOT/private/send.py" --credential-file "$callback" || true
        exit "$rc"
    }
    trap on_exit EXIT
fi
mkdir -p "$CAMPAIGN_ROOT/cache/$mode-$arm" "$WANDB_DIR" "$CAMPAIGN_ROOT/allocations"
scontrol show job "$SLURM_JOB_ID" > "$CAMPAIGN_ROOT/allocations/$SLURM_JOB_ID.txt"
mounts="$CAMPAIGN_ROOT/source/src:/workspace/src,$CAMPAIGN_ROOT/native:/workspace/native,$CAMPAIGN_ROOT:$CAMPAIGN_ROOT,$DATA_ROOT:$DATA_ROOT,$CAMPAIGN_ROOT/cache/$mode-$arm:/workspace/.data_cache"
srun --kill-on-bad-exit=1 --container-image="$SAMUDRA_CONTAINER_IMAGE_REF" --container-mount-home --container-mounts="$mounts" bash -c '
set -euo pipefail
export PYTHONPATH=/workspace/src:/workspace/native
cd "$CAMPAIGN_ROOT/source/experiments/score-trained-samudra-1deg"
/workspace/.venv/bin/python -c "import torch,samudra,samudra_rust_loader; print(torch.__version__,samudra.__file__,samudra_rust_loader.__file__)"
args=()
if [[ "$2" == qualification ]]; then
    export WANDB_MODE=disabled
    args+=(--qualify)
    if [[ "$1" == D ]]; then args+=(--compute-scales); fi
fi
exec /workspace/.venv/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 train_score.py "$CAMPAIGN_ROOT/configs/$2-$1.yaml" "${args[@]}"
' bash "$arm" "$mode"
