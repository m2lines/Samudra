#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --account=torch_pr_347_lzanna
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --constraint=h100
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --no-requeue

set -euo pipefail
: "${CAMPAIGN_ROOT:?Set campaign directory}"
: "${SOURCE_ROOT:?Set immutable source directory}"
: "${DATA_ROOT:?Set existing OM4 data directory}"
: "${SIF_PATH:?Set verified container path}"
: "${FOLLOWUP_CREDENTIAL:?Set private callback credential path}"
arm="$1"
phase="$2"
case "$arm" in F) mode=forecast ;; M) mode=multitask ;; *) exit 2 ;; esac
case "$phase" in qualification|full|selected|secondary-qualification|continuous-qualification|continuous|completion-qualification|completion) ;; *) exit 2 ;; esac
export EVAL_ARM="$arm"
export EVAL_PHASE="$phase"
export EVAL_OUTPUT="$CAMPAIGN_ROOT/outputs/eval-$phase-$arm-$SLURM_JOB_ID"
export EVAL_CHECKPOINT="$CAMPAIGN_ROOT/outputs/train-$mode-rtx/saved_nets/ema_ckpt.pt"
if [[ "$phase" == selected ]]; then
    export EVAL_CHECKPOINT="$CAMPAIGN_ROOT/outputs/train-$mode-rtx/saved_nets/best_validation_ckpt.pt"
fi
# Completion is delivered by a separate CPU-only afterany job.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 WANDB_MODE=online
export NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export SAMUDRA_CODE_COMMIT="$(cat "$SOURCE_ROOT/COMMIT")"
export SAMUDRA_CODE_REPO_URL=https://github.com/m2lines/Samudra.git
export WANDB_GIT_COMMIT="$SAMUDRA_CODE_COMMIT"
export WANDB_GIT_REMOTE_URL="$SAMUDRA_CODE_REPO_URL"
export SAMUDRA_CONTAINER_IMAGE_REF="$SIF_PATH"
export WANDB_DIR="$EVAL_OUTPUT"
export NPROC="${NPROC:-2}"
export CONFIG_PATH="${CONFIG_PATH:-$CAMPAIGN_ROOT/configs/train-$mode-rtx.yaml}"
export APPTAINER_CACHEDIR=/scratch/jr7309/apptainer-cache
export APPTAINER_TMPDIR="${SLURM_TMPDIR:-/tmp}/apptainer-$SLURM_JOB_ID"
mkdir -p "$APPTAINER_TMPDIR" "$WANDB_DIR" "$CAMPAIGN_ROOT/cache/$phase-$mode" "$CAMPAIGN_ROOT/allocations"
scontrol show job "$SLURM_JOB_ID" > "$CAMPAIGN_ROOT/allocations/$SLURM_JOB_ID-${SLURM_RESTART_COUNT:-0}.txt"
python3 "$CAMPAIGN_ROOT/private/send.py" --credential-file "$FOLLOWUP_CREDENTIAL" --check
export PYTHONPATH="$SOURCE_ROOT/src"
: "${CUDA_VISIBLE_DEVICES:?Slurm must define allocated GPU visibility}"
for name in OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS PYTHONUNBUFFERED PYTHONNOUSERSITE WANDB_MODE WANDB_API_KEY WANDB_DIR WANDB_GIT_COMMIT WANDB_GIT_REMOTE_URL CUDA_VISIBLE_DEVICES NCCL_DEBUG TORCH_NCCL_ASYNC_ERROR_HANDLING SAMUDRA_CODE_COMMIT SAMUDRA_CODE_REPO_URL SAMUDRA_CONTAINER_IMAGE_REF PYTHONPATH NPROC SOURCE_ROOT CAMPAIGN_ROOT CONFIG_PATH EVAL_ARM EVAL_PHASE EVAL_OUTPUT EVAL_CHECKPOINT SLURM_JOB_ID; do
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
if [[ "$EVAL_PHASE" == secondary-qualification ]]; then
    selected="$(dirname "$EVAL_CHECKPOINT")/best_validation_ckpt.pt"
    /workspace/.venv/bin/python evaluate_score.py "$CONFIG_PATH" "$selected" "$EVAL_OUTPUT/selected" --arm "$EVAL_ARM" --members 16 --checkpoint-role validation-selected
    /workspace/.venv/bin/python evaluate_completion.py "$CONFIG_PATH" "$EVAL_CHECKPOINT" "$EVAL_OUTPUT/completion" --arm "$EVAL_ARM" --origin-limit 2
    exec /workspace/.venv/bin/python diagnose_score.py "$CAMPAIGN_ROOT" "$EVAL_OUTPUT/continuous" --arm "$EVAL_ARM" --max-steps 2
fi
if [[ "$EVAL_PHASE" == continuous* ]]; then
    args=()
    if [[ "$EVAL_PHASE" == continuous-qualification ]]; then args+=(--max-steps 2); fi
    exec /workspace/.venv/bin/python diagnose_score.py "$CAMPAIGN_ROOT" "$EVAL_OUTPUT" --arm "$EVAL_ARM" "${args[@]}"
fi
if [[ "$EVAL_PHASE" == completion* ]]; then
    args=()
    if [[ "$EVAL_PHASE" == completion-qualification ]]; then args+=(--origin-limit 2); fi
    exec /workspace/.venv/bin/python evaluate_completion.py "$CONFIG_PATH" "$EVAL_CHECKPOINT" "$EVAL_OUTPUT" --arm "$EVAL_ARM" "${args[@]}"
fi
args=()
if [[ "$EVAL_PHASE" == selected ]]; then args+=(--checkpoint-role validation-selected); fi
if [[ "$EVAL_PHASE" == qualification ]]; then args+=(--origin-limit 2); fi
launcher=()
if [[ "$NPROC" != 1 ]]; then
    launcher=(-m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="$NPROC")
fi
exec /workspace/.venv/bin/python "${launcher[@]}" evaluate_score.py "$CONFIG_PATH" "$EVAL_CHECKPOINT" "$EVAL_OUTPUT" --arm "$EVAL_ARM" --members 16 "${args[@]}"
' bash "$mode" "$phase" &
wait "$!"
