#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
: "${CODE_COMMIT:?Set the pushed full experiment commit}" "${WANDB_API_KEY:?}"
export BUNDLE_DIR=/scratch/am16581/runs/_experiments/2026-11-09-onedeg-tendency-residual
export BASE_NAME=2026-11-09-onedeg-tendency-residual
export DATA_ROOT=/scratch/am16581/data/om4_onedeg_v3 OUTPUT_BASE=/scratch/am16581/runs
export SCRATCH_DIR=/scratch/am16581 SIF_DIR=/scratch/am16581/.apptainer-images
export SIF_PATH=/scratch/am16581/.apptainer-images/physicsnemo-26.05-4f86be45.sif
export CODE_LAYER="/scratch/am16581/.apptainer-code-layers/samudra-code-${CODE_COMMIT}.img"
export IMAGE_REF=ghcr.io/m2lines/ocean-emulator-physicsnemo:26.05-4f86be45a09cb9eba1b51b6273e0189941b02495
export APPTAINER_CACHEDIR=/scratch/am16581/apptainer-cache
export SINGULARITY_CACHEDIR=/scratch/am16581/singularity-cache
export NCCL_P2P_DISABLE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export REQUEUE_ON_USR1=1 GPUS_PER_NODE=8 WANDB_MODE=online
test ! -e "${BUNDLE_DIR}/jobs.tsv"
test ! -e "${OUTPUT_BASE}/${BASE_NAME}"
test ! -e "${OUTPUT_BASE}/${BASE_NAME}-2in1out"
(cd "${BUNDLE_DIR}" && sha256sum --check --status bundle.sha256)
common=(--parsable --account=torch_pr_347_lzanna --nodes=1 --ntasks=1 --chdir=/scratch/am16581 --kill-on-invalid-dep=yes)
submit_phase() {
  local phase="$1" name="$2" train_name="$3"
  shift 3
  export PHASE="${phase}" NAME="${name}"
  export TRAIN_NAME="${train_name}" CONFIG_SUFFIX=""
  if [[ "${train_name}" == *-2in1out ]]; then
    export CONFIG_SUFFIX=_2in1out
  fi
  local job
  job=$(sbatch "${common[@]}" --job-name="tendency-${phase}" \
    --output="${BUNDLE_DIR}/logs/${phase}-%j.out" --error="${BUNDLE_DIR}/logs/${phase}-%j.err" \
    "$@" "${BUNDLE_DIR}/phase.sbatch")
  printf '%s\t%s\t%s\n' "${phase}" "${job}" "${name}" >> "${BUNDLE_DIR}/jobs.tsv"
  printf '%s' "${job}"
}
mkdir -p "${BUNDLE_DIR}/logs"
prep=$(submit_phase prepare "${BASE_NAME}-prepare" "${BASE_NAME}" --partition=cs --cpus-per-task=8 --mem=32G --time=02:00:00)
train=$(submit_phase train "${BASE_NAME}" "${BASE_NAME}" --partition=rtx6000_lzanna --cpus-per-task=128 --mem=1400G \
  --gres=gpu:rtx6000:8 --time=24:00:00 --requeue --signal=B:USR1@300 --dependency="afterok:${prep}")
train_one=$(submit_phase train "${BASE_NAME}-2in1out" "${BASE_NAME}-2in1out" --partition=rtx6000_lzanna --cpus-per-task=128 --mem=1400G \
  --gres=gpu:rtx6000:8 --time=24:00:00 --requeue --signal=B:USR1@300 --dependency="afterok:${prep}")
eval_job=$(submit_phase eval "${BASE_NAME}-eval" "${BASE_NAME}" --partition=rtx6000_lzanna --cpus-per-task=16 --mem=175G \
  --gres=gpu:rtx6000:1 --time=04:00:00 --dependency="afterok:${train}")
eval_one=$(submit_phase eval "${BASE_NAME}-2in1out-eval" "${BASE_NAME}-2in1out" --partition=rtx6000_lzanna --cpus-per-task=16 --mem=175G \
  --gres=gpu:rtx6000:1 --time=04:00:00 --dependency="afterok:${train_one}")
submit_phase viz "${BASE_NAME}-viz" "${BASE_NAME}" --partition=cs --cpus-per-task=32 --mem=256G \
  --time=24:00:00 --dependency="afterok:${eval_job}:${eval_one}" > /dev/null
submit_phase metrics "${BASE_NAME}-metrics" "${BASE_NAME}" --partition=cs --cpus-per-task=16 --mem=128G \
  --time=12:00:00 --dependency="afterok:${eval_job}:${eval_one}" > /dev/null
submit_phase imprint "${BASE_NAME}-imprint" "${BASE_NAME}" --partition=cs --cpus-per-task=16 --mem=64G \
  --time=12:00:00 --dependency="afterok:${eval_job}:${eval_one}" > /dev/null
cat "${BUNDLE_DIR}/jobs.tsv"
