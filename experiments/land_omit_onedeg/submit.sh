#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
#
# Train Samudra 2 from scratch at 1° with land: mask and land: omit, then roll
# each out for 8 years and score it with samudra.metrics.
#
#   CODE_COMMIT=<pushed full sha> bash experiments/land_omit_onedeg/submit.sh [all|mask|omit]
#
# Run on the Torch login node. Needs WANDB_API_KEY (from ~/.bashrc), the
# harness scripts in $HOME (see docs/torch.md), and a code layer for
# CODE_COMMIT built by scripts/build_apptainer_code_layer.sh.
set -euo pipefail
: "${CODE_COMMIT:?Set the pushed full commit of experiment/land-omit-onedeg}"
: "${WANDB_API_KEY:?}"
WHAT="${1:-all}"

U=/scratch/am16581
export SCRATCH_DIR=$U SIF_DIR=$U/.apptainer-images OUTPUT_BASE=$U/runs
export DATA_ROOT=$U/data/v2026-09/om4_onedeg_rust
export APPTAINER_CACHEDIR=$U/apptainer-cache SINGULARITY_CACHEDIR=$U/singularity-cache
export NCCL_P2P_DISABLE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 WANDB_MODE=online
export PATH=/share/apps/apptainer/bin:/opt/slurm/bin:$PATH
export SIF_PATH=$SIF_DIR/physicsnemo-26.05-c473ebd4.sif
export CODE_LAYER=$U/.apptainer-code-layers/samudra-code-$CODE_COMMIT.img
test -e "$CODE_LAYER" && test -e "$SIF_PATH" && test -e "$DATA_ROOT/OM4.zarr/hfds_anomalies"

DATE=2026-11-15
CFG=src/samudra/configs/experiments/land_omit_onedeg
JOBS=$U/agent-logs/$DATE-land-omit-onedeg-jobs.tsv
W="--experiment.wandb.entity=ocean_emulators --experiment.wandb.project=default --experiment.wandb.group=land-omit-onedeg-20261115"
common=(--parsable --account=torch_pr_347_lzanna --nodes=1 --ntasks-per-node=1
  --chdir=$U --kill-on-invalid-dep=yes)

record() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$CODE_COMMIT" >> "$JOBS"; echo "$1 $2 $3"; }

metrics_job() {  # name config dependency
  local name=$1 config=$2 dep=$3
  sbatch "${common[@]}" --partition=cs --cpus-per-task=16 --mem=128G --time=12:00:00 \
    --job-name="metrics-$name" --dependency="afterok:$dep" \
    --output="$U/agent-logs/$name-metrics-%j.out" \
    --wrap "set -euo pipefail; export PATH=$PATH; \
      export APPTAINERENV_PYTHONPATH=/opt/samudra-code/src APPTAINERENV_MPLBACKEND=Agg \
        APPTAINERENV_PYTHONUNBUFFERED=1; \
      apptainer exec --overlay $CODE_LAYER:ro --bind $U:$U --pwd /opt/samudra-code $SIF_PATH \
        /workspace/.venv/bin/python -m samudra.metrics $config \
        --experiment.name=$name --experiment.data_root=$DATA_ROOT \
        --experiment.base_output_dir=$OUTPUT_BASE --experiment.wandb.mode=disabled"
}

submit_arm() {  # land
  local land=$1
  local name=$DATE-land-$land-onedeg
  test ! -e "$OUTPUT_BASE/$name"

  export NAME=$name CONFIG=$CFG/train_$land.yaml GPUS_PER_NODE=8 REQUEUE_ON_USR1=1
  export DATA_CACHE_DIR=$U/.data_cache/$name
  export ARGS="$W"
  unset TARGET_CHECKPOINT
  local train
  train=$(sbatch "${common[@]}" --partition=rtx6000_lzanna --cpus-per-task=128 --mem=1400G \
    --gres=gpu:rtx6000:8 --time=24:00:00 --requeue --signal=B:USR1@300 \
    --job-name=land-$land-train --output="$U/agent-logs/$name-train-%j.out" \
    --error="$U/agent-logs/$name-train-%j.err" ~/slurm_apptainer_train.sbatch)
  record train "$train" "$name"

  export NAME=$name-eval CONFIG=$CFG/eval_$land.yaml TARGET_CHECKPOINT=$name/saved_nets/ema_ckpt.pt
  export ARGS="$W"
  unset GPUS_PER_NODE REQUEUE_ON_USR1
  local eval_job
  eval_job=$(sbatch "${common[@]}" --partition=rtx6000_lzanna --cpus-per-task=16 --mem=175G \
    --gres=gpu:rtx6000:1 --time=04:00:00 --dependency="afterok:$train" \
    --job-name=land-$land-eval --output="$U/agent-logs/$name-eval-%j.out" \
    --error="$U/agent-logs/$name-eval-%j.err" ~/slurm_apptainer_eval.sbatch)
  record eval "$eval_job" "$name-eval"
  record metrics "$(metrics_job "$name-eval" "/opt/samudra-code/$CFG/eval_$land.yaml" "$eval_job")" "$name-eval"
}

if [[ "$WHAT" == all || "$WHAT" == mask ]]; then submit_arm mask; fi
if [[ "$WHAT" == all || "$WHAT" == omit ]]; then submit_arm omit; fi
