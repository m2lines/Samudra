#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
#
# Submit the normalized_delta 2-in-1-out run on Torch, plus a re-evaluation of
# PR #969's final EMA checkpoint on the same v2026-09 data.
#
#   CODE_COMMIT=<pushed full sha> bash experiments/normalized_delta_2in1out/submit.sh [all|nd|pr969]
#
# Run on the Torch login node. Needs WANDB_API_KEY (from ~/.bashrc) and the
# harness scripts in $HOME (see docs/torch.md).
set -euo pipefail
: "${CODE_COMMIT:?Set the pushed full commit of experiment/normalized-delta-2in1out}"
: "${WANDB_API_KEY:?}"
WHAT="${1:-all}"

U=/scratch/am16581
export SCRATCH_DIR=$U SIF_DIR=$U/.apptainer-images OUTPUT_BASE=$U/runs
export DATA_ROOT=$U/data/v2026-09/om4_onedeg
export APPTAINER_CACHEDIR=$U/apptainer-cache SINGULARITY_CACHEDIR=$U/singularity-cache
export NCCL_P2P_DISABLE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 WANDB_MODE=online
export PATH=/share/apps/apptainer/bin:/opt/slurm/bin:$PATH

DATE=2026-11-13
ND_NAME=$DATE-normalized-delta-2in1out
PR969_TRAIN=2026-11-09-onedeg-tendency-scaled-2in1out-rust-rtx8
PR969_NAME=$PR969_TRAIN-eval-v2026-09
JOBS=$U/agent-logs/$ND_NAME-jobs.tsv
W="--experiment.wandb.entity=ocean_emulators --experiment.wandb.project=default --experiment.wandb.group=normalized-delta-vs-969-20261113"
common=(--parsable --account=torch_pr_347_lzanna --nodes=1 --ntasks-per-node=1
  --chdir=$U --kill-on-invalid-dep=yes)

record() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$JOBS"; echo "$1 $2 $3"; }

# Run `python -m samudra.metrics` on a finished eval, inside the container.
metrics_job() {  # name sif layer config dependency
  local name=$1 sif=$2 layer=$3 config=$4 dep=$5
  sbatch "${common[@]}" --partition=cs --cpus-per-task=16 --mem=128G --time=12:00:00 \
    --job-name="metrics-$name" --dependency="afterok:$dep" \
    --output="$U/agent-logs/$name-metrics-%j.out" \
    --wrap "set -euo pipefail; export PATH=$PATH; \
      export APPTAINERENV_PYTHONPATH=/opt/samudra-code/src APPTAINERENV_MPLBACKEND=Agg \
        APPTAINERENV_PYTHONUNBUFFERED=1; \
      apptainer exec --overlay $layer:ro --bind $U:$U --pwd /opt/samudra-code $sif \
        /workspace/.venv/bin/python -m samudra.metrics $config \
        --experiment.name=$name --experiment.data_root=$DATA_ROOT \
        --experiment.base_output_dir=$OUTPUT_BASE --experiment.wandb.mode=disabled"
}

if [[ "$WHAT" == all || "$WHAT" == nd ]]; then
  export SIF_PATH=$SIF_DIR/physicsnemo-26.05-c473ebd4.sif
  export CODE_LAYER=$U/.apptainer-code-layers/samudra-code-$CODE_COMMIT.img
  test -e "$CODE_LAYER" && test -e "$SIF_PATH"
  test -e "$DATA_ROOT/OM4_increment_rms_wet.zarr"
  test ! -e "$OUTPUT_BASE/$ND_NAME"
  CFG=experiments/normalized_delta_2in1out

  export NAME=$ND_NAME CONFIG=$CFG/train.yaml GPUS_PER_NODE=8 REQUEUE_ON_USR1=1
  export DATA_CACHE_DIR=$U/.data_cache/$ND_NAME
  export ARGS="$W"
  train=$(sbatch "${common[@]}" --partition=rtx6000_lzanna --cpus-per-task=128 --mem=1400G \
    --gres=gpu:rtx6000:8 --time=24:00:00 --requeue --signal=B:USR1@300 \
    --job-name=nd-train --output="$U/agent-logs/$ND_NAME-train-%j.out" \
    --error="$U/agent-logs/$ND_NAME-train-%j.err" ~/slurm_apptainer_train.sbatch)
  record train "$train" "$ND_NAME"

  export NAME=$ND_NAME-eval CONFIG=$CFG/eval.yaml TARGET_CHECKPOINT=$ND_NAME/saved_nets/ema_ckpt.pt
  export ARGS="$W"
  unset GPUS_PER_NODE REQUEUE_ON_USR1
  eval_job=$(sbatch "${common[@]}" --partition=rtx6000_lzanna --cpus-per-task=16 --mem=175G \
    --gres=gpu:rtx6000:1 --time=04:00:00 --dependency="afterok:$train" \
    --job-name=nd-eval --output="$U/agent-logs/$ND_NAME-eval-%j.out" \
    --error="$U/agent-logs/$ND_NAME-eval-%j.err" ~/slurm_apptainer_eval.sbatch)
  record eval "$eval_job" "$ND_NAME-eval"
  record metrics "$(metrics_job "$ND_NAME-eval" "$SIF_PATH" "$CODE_LAYER" \
    "/opt/samudra-code/$CFG/eval.yaml" "$eval_job")" "$ND_NAME-eval"
fi

if [[ "$WHAT" == all || "$WHAT" == pr969 ]]; then
  # #969's own code, container and configs; only the data root and name change.
  # Its v3 and v2026-09 normalization statistics are identical.
  # The finished #969 run (rust-rtx8) trained from 6574fbb6.
  C969=6574fbb6f060c672a9ce94b07d844983da42d265
  export SIF_PATH=$SIF_DIR/physicsnemo-26.05-4f86be45.sif
  export CODE_LAYER=$U/.apptainer-code-layers/samudra-code-$C969.img
  test -e "$CODE_LAYER" && test ! -e "$OUTPUT_BASE/$PR969_NAME"
  CFG969=src/samudra/configs/experiments/onedeg_tendency_scaled
  export NAME=$PR969_NAME CONFIG=$CFG969/eval.yaml
  export CKPT_PATH=$OUTPUT_BASE/$PR969_TRAIN/saved_nets/ema_ckpt.pt
  unset TARGET_CHECKPOINT
  export ARGS="$W"
  eval969=$(sbatch "${common[@]}" --partition=rtx6000_lzanna --cpus-per-task=16 --mem=175G \
    --gres=gpu:rtx6000:1 --time=04:00:00 --job-name=pr969-eval \
    --output="$U/agent-logs/$PR969_NAME-%j.out" --error="$U/agent-logs/$PR969_NAME-%j.err" \
    ~/slurm_apptainer_eval.sbatch)
  record pr969-eval "$eval969" "$PR969_NAME"
  record pr969-metrics "$(metrics_job "$PR969_NAME" "$SIF_PATH" "$CODE_LAYER" \
    "/opt/samudra-code/$CFG969/metrics.yaml" "$eval969")" "$PR969_NAME"
fi
