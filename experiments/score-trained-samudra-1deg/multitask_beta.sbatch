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
#SBATCH --time=18:00:00

set -euo pipefail
: "${CAMPAIGN_ROOT:?Set campaign directory}"
: "${SOURCE_ROOT:?Set immutable source directory}"
: "${DATA_ROOT:?Set existing OM4 data directory}"
: "${NATIVE_ROOT:?Set existing ARM Rust loader directory}"
: "${FOLLOWUP_CREDENTIAL:?Set private callback credential path}"
mode="$1"
phase="$2"
case "$mode" in forecast|multitask) ;; *) exit 2 ;; esac
case "$phase" in qualification|train) ;; *) exit 2 ;; esac
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
export SAMUDRA_CODE_REPO_URL=https://github.com/m2lines/Samudra.git
export SAMUDRA_CONTAINER_IMAGE_REF=ghcr.io/m2lines/ocean-emulator-physicsnemo@sha256:7d78e8fb9592abdf046d88e55004f15cfe34f0c8ee38b17d83e98cc05b6f82c4
export WANDB_DIR="$CAMPAIGN_ROOT/wandb/$phase-$mode"
export NPROC="${NPROC:-4}"
mkdir -p "$WANDB_DIR" "$CAMPAIGN_ROOT/cache/$phase-$mode" "$CAMPAIGN_ROOT/allocations"
scontrol show job "$SLURM_JOB_ID" > "$CAMPAIGN_ROOT/allocations/$SLURM_JOB_ID.txt"
python3 "$CAMPAIGN_ROOT/private/send.py" --credential-file "$FOLLOWUP_CREDENTIAL" --check
mounts="$SOURCE_ROOT/src:/workspace/src,$NATIVE_ROOT:/workspace/native,$CAMPAIGN_ROOT:$CAMPAIGN_ROOT,$DATA_ROOT:$DATA_ROOT,$CAMPAIGN_ROOT/cache/$phase-$mode:/workspace/.data_cache"
srun --input=none --kill-on-bad-exit=1 --container-image="$SAMUDRA_CONTAINER_IMAGE_REF" --container-mount-home --container-mounts="$mounts" bash -c '
set -euo pipefail
export PYTHONPATH=/workspace/src:/workspace/native
cd "$SOURCE_ROOT/experiments/score-trained-samudra-1deg"
/workspace/.venv/bin/python -c "import torch,samudra,samudra_rust_loader; print(torch.__version__,samudra.__file__,samudra_rust_loader.__file__)"
args=()
if [[ "$2" == qualification ]]; then export WANDB_MODE=disabled; args+=(--qualify); fi
exec /workspace/.venv/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="$NPROC" train_multitask.py "$CAMPAIGN_ROOT/configs/$2-$1.yaml" "${args[@]}"
' bash "$mode" "$phase"
