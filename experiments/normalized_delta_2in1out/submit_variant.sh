#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
#
# Submit train -> eval -> metrics for one variant of PR #984 on Torch.
#
#   CODE_COMMIT=<pushed full sha> CFG=experiments/<variant> NAME=<run name> \
#     [AFTER=<slurm job id to wait for>] [EXTRA_ARGS="--resume_ckpt_path=..."] \
#     bash submit_variant.sh
#
# Training runs in the preemptible pool (account torch_pr_500_lzanna) on one
# node with 4 RTX6000 GPUs and 2 gradient-accumulation steps, so the global
# batch and the per-GPU BatchNorm batch match 8 GPUs without accumulation. The
# lab's rtx6000_lzanna partition is capped at 8 GPUs and 128 CPUs for the
# whole group, so it is left for evaluation. Preempted jobs requeue and resume
# from their latest checkpoint (preemptible: true), except fine-tunes, which
# restart.
#
# CFG must contain train.yaml and eval.yaml. Run on the Torch login node with
# WANDB_API_KEY set (from ~/.bashrc) and the harness scripts in $HOME.
#
# Code layers carry only src/, so the experiments/ directory at CODE_COMMIT is
# snapshotted under $OUTPUT_BASE/_experiments/$NAME/ (bound into the
# container) and configs are read from there.
set -euo pipefail
: "${CODE_COMMIT:?}" "${CFG:?}" "${NAME:?}" "${WANDB_API_KEY:?}"

U=/scratch/am16581
export SCRATCH_DIR=$U SIF_DIR=$U/.apptainer-images OUTPUT_BASE=$U/runs
# v2026-09 plus a materialized hfds_anomalies for the Rust loader
# (stage_rust_store.py); all other arrays are symlinks to om4_onedeg.
export DATA_ROOT=$U/data/v2026-09/om4_onedeg_rust
export APPTAINER_CACHEDIR=$U/apptainer-cache SINGULARITY_CACHEDIR=$U/singularity-cache
export NCCL_P2P_DISABLE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 WANDB_MODE=online
export PATH=/share/apps/apptainer/bin:/opt/slurm/bin:$PATH
export SIF_PATH=$SIF_DIR/physicsnemo-26.05-c473ebd4.sif
export CODE_LAYER=$U/.apptainer-code-layers/samudra-code-$CODE_COMMIT.img
JOBS=$U/agent-logs/$NAME-jobs.tsv
W="--experiment.wandb.entity=ocean_emulators --experiment.wandb.project=default --experiment.wandb.group=normalized-delta-vs-969-20261113"
common=(--parsable --account=torch_pr_347_lzanna --nodes=1 --ntasks-per-node=1
  --chdir=$U --kill-on-invalid-dep=yes)

test -e "$CODE_LAYER" && test -e "$SIF_PATH"
test -e "$DATA_ROOT/OM4_increment_rms_wet.zarr" && test -e "$DATA_ROOT/verification.json"
test ! -e "$OUTPUT_BASE/$NAME"

SNAP=$OUTPUT_BASE/_experiments/$NAME
test ! -e "$SNAP"
tmp=$(mktemp -d)
git -C "$tmp" init -q
git -C "$tmp" fetch -q --depth 1 https://github.com/m2lines/Samudra.git "$CODE_COMMIT"
mkdir -p "$SNAP"
git -C "$tmp" archive FETCH_HEAD experiments | tar -x -C "$SNAP"
rm -rf "$tmp"
echo "$CODE_COMMIT" > "$SNAP/CODE_COMMIT"
(cd "$SNAP" && find experiments -type f -print0 | sort -z | xargs -0 sha256sum > configs.sha256)
CFG_DIR=$SNAP/$CFG
test -f "$CFG_DIR/train.yaml" && test -f "$CFG_DIR/eval.yaml"
record() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$JOBS"; echo "$1 $2 $3"; }

dep=()
[[ -n "${AFTER:-}" ]] && dep=(--dependency="afterok:$AFTER")
export CONFIG=$CFG_DIR/train.yaml GPUS_PER_NODE=4 REQUEUE_ON_USR1=1
export DATA_CACHE_DIR=$U/.data_cache/$NAME
export ARGS="$W --gradient_accumulation_steps=${ACCUM:-2} ${EXTRA_ARGS:-}"
pool=(--parsable --account=torch_pr_500_lzanna --nodes=1 --ntasks-per-node=1
  --chdir=$U --kill-on-invalid-dep=yes --constraint=rtx6000 --gres=gpu:4
  --cpus-per-task=32 --mem=256G "--comment=preemption=yes;requeue=true")
train=$(NAME=$NAME sbatch "${pool[@]}" "${dep[@]}" --time=24:00:00 \
  --requeue --signal=B:USR1@300 --job-name="${NAME: -24}" \
  --output="$U/agent-logs/$NAME-train-%j.out" --error="$U/agent-logs/$NAME-train-%j.err" \
  ~/slurm_apptainer_train.sbatch)
record train "$train" "$NAME"

unset GPUS_PER_NODE REQUEUE_ON_USR1
export CONFIG=$CFG_DIR/eval.yaml TARGET_CHECKPOINT=$NAME/saved_nets/ema_ckpt.pt ARGS="$W"
eval_job=$(NAME=$NAME-eval sbatch "${common[@]}" --partition=rtx6000_lzanna \
  --cpus-per-task=16 --mem=175G --gres=gpu:rtx6000:1 --time=04:00:00 \
  --dependency="afterok:$train" --job-name=eval-"${NAME: -19}" \
  --output="$U/agent-logs/$NAME-eval-%j.out" --error="$U/agent-logs/$NAME-eval-%j.err" \
  ~/slurm_apptainer_eval.sbatch)
record eval "$eval_job" "$NAME-eval"

metrics=$(sbatch "${common[@]}" --partition=cs --cpus-per-task=16 --mem=128G --time=12:00:00 \
  --job-name=metrics-"${NAME: -16}" --dependency="afterok:$eval_job" \
  --output="$U/agent-logs/$NAME-metrics-%j.out" \
  --wrap "set -euo pipefail; export PATH=$PATH; \
    export APPTAINERENV_PYTHONPATH=/opt/samudra-code/src APPTAINERENV_MPLBACKEND=Agg \
      APPTAINERENV_PYTHONUNBUFFERED=1; \
    apptainer exec --overlay $CODE_LAYER:ro --bind $U:$U --pwd /opt/samudra-code $SIF_PATH \
      /workspace/.venv/bin/python -m samudra.metrics $CFG_DIR/eval.yaml \
      --experiment.name=$NAME-eval --experiment.data_root=$DATA_ROOT \
      --experiment.base_output_dir=$OUTPUT_BASE --experiment.wandb.mode=disabled")
record metrics "$metrics" "$NAME-eval"
