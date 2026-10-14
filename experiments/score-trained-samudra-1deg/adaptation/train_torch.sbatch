#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --account=torch_pr_347_lzanna
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --partition=rtx6000_lzanna
#SBATCH --constraint=rtx6000
#SBATCH --mem=192G
#SBATCH --time=14:00:00
#SBATCH --no-requeue

set -euo pipefail
: "${CAMPAIGN_ROOT:?Set campaign directory}"
: "${SOURCE_ROOT:?Set immutable source directory}"
: "${DATA_ROOT:?Set existing OM4 data directory}"
: "${SIF_PATH:?Set verified container path}"
: "${FOLLOWUP_CREDENTIAL:?Set private callback credential path}"
mode="$1"
phase="$2"
case "$mode" in forecast|multitask) ;; *) exit 2 ;; esac
case "$phase" in qualification|train) ;; *) exit 2 ;; esac
# Finetuning starts a fresh optimizer; recover interruptions explicitly from its own checkpoint.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 WANDB_MODE=online
export NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_P2P_DISABLE=1
export SAMUDRA_CODE_COMMIT="$(cat "$SOURCE_ROOT/COMMIT")"
export SAMUDRA_CODE_REPO_URL=https://github.com/m2lines/Samudra.git
export WANDB_GIT_COMMIT="$SAMUDRA_CODE_COMMIT"
export WANDB_GIT_REMOTE_URL="$SAMUDRA_CODE_REPO_URL"
export SAMUDRA_CONTAINER_IMAGE_REF="$SIF_PATH"
export WANDB_DIR="$CAMPAIGN_ROOT/wandb/$phase-$mode"
export NPROC="${NPROC:-2}"
export CONFIG_PATH="${CONFIG_PATH:-$CAMPAIGN_ROOT/configs/$phase-$mode.yaml}"
export APPTAINER_CACHEDIR=/scratch/jr7309/apptainer-cache
export APPTAINER_TMPDIR="${SLURM_TMPDIR:-/tmp}/apptainer-$SLURM_JOB_ID"
mkdir -p "$APPTAINER_TMPDIR" "$WANDB_DIR" "$CAMPAIGN_ROOT/cache/$phase-$mode" "$CAMPAIGN_ROOT/allocations"
scontrol show job "$SLURM_JOB_ID" > "$CAMPAIGN_ROOT/allocations/$SLURM_JOB_ID-${SLURM_RESTART_COUNT:-0}.txt"
python3 "$CAMPAIGN_ROOT/private/send.py" --credential-file "$FOLLOWUP_CREDENTIAL" --check
export PYTHONPATH="$SOURCE_ROOT/src"
: "${CUDA_VISIBLE_DEVICES:?Slurm must define allocated GPU visibility}"
for name in OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS PYTHONUNBUFFERED PYTHONNOUSERSITE WANDB_MODE WANDB_API_KEY WANDB_DIR WANDB_GIT_COMMIT WANDB_GIT_REMOTE_URL CUDA_VISIBLE_DEVICES NCCL_DEBUG NCCL_P2P_DISABLE TORCH_NCCL_ASYNC_ERROR_HANDLING SAMUDRA_CODE_COMMIT SAMUDRA_CODE_REPO_URL SAMUDRA_CONTAINER_IMAGE_REF PYTHONPATH NPROC SOURCE_ROOT CAMPAIGN_ROOT CONFIG_PATH SLURM_JOB_ID; do
    export "APPTAINERENV_$name=${!name}"
done
apptainer exec --nv --cleanenv \
    --bind "$CAMPAIGN_ROOT:$CAMPAIGN_ROOT,$SOURCE_ROOT:$SOURCE_ROOT:ro,$DATA_ROOT:$DATA_ROOT:ro,$CAMPAIGN_ROOT/cache/$phase-$mode:/workspace/.data_cache" \
    "$SIF_PATH" bash -c '
set -euo pipefail
cmp "$SOURCE_ROOT/uv.lock" /workspace/uv.lock
cmp "$SOURCE_ROOT/pyproject.toml" /workspace/pyproject.toml
cd "$SOURCE_ROOT/experiments/score-trained-samudra-1deg"
/workspace/.venv/bin/python -c "import os,torch,samudra,samudra_rust_loader; print(torch.__version__,samudra.__file__,samudra_rust_loader.__file__); print(torch.cuda.get_device_name(),torch.cuda.device_count()); assert torch.cuda.device_count() == int(os.environ[\"NPROC\"])"
args=()
if [[ "$2" == qualification ]]; then export WANDB_MODE=disabled; args+=(--qualify); fi
exec /workspace/.venv/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="$NPROC" train_multitask.py "$CONFIG_PATH" "${args[@]}"
' bash "$mode" "$phase" &
wait "$!"
