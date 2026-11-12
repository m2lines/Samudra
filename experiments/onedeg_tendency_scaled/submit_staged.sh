#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
: "${SCHEDULING_DIR:?}" "${WANDB_API_KEY:?}"
(cd "${SCHEDULING_DIR}" && sha256sum --check --status scheduling.sha256)
test ! -e "${SCHEDULING_DIR}/jobs.tsv"
base_name=2026-11-09-onedeg-tendency-scaled-2in1out
test ! -e "/scratch/am16581/runs/${base_name}"
test ! -e "/scratch/am16581/runs/${base_name}-stage4gpu"
common=(--parsable --account=torch_pr_500_lzanna --ntasks-per-node=1 --chdir=/scratch/am16581
  --gres=gpu:4 --constraint=rtx6000 --cpus-per-task=32 --mem=256G
  --comment=preemption=yes\;requeue=true --requeue --signal=B:USR1@300 --kill-on-invalid-dep=yes)
submit_stage() {
  export STAGE="$1"
  shift
  local job
  job=$(sbatch "${common[@]}" --job-name="scaled-${STAGE}" \
    --output="${SCHEDULING_DIR}/${STAGE}-%j.out" --error="${SCHEDULING_DIR}/${STAGE}-%j.err" \
    "$@" "${SCHEDULING_DIR}/staged_job.sbatch")
  printf '%s\t%s\n' "${STAGE}" "${job}" >> "${SCHEDULING_DIR}/jobs.tsv"
  printf '%s' "${job}"
}
four=$(submit_stage 4gpu --nodes=1 --time=04:00:00)
interim=$(submit_stage interim-eval --nodes=1 --gres=gpu:1 --cpus-per-task=16 --mem=175G \
  --time=04:00:00 --comment= --dependency="afterok:${four}")
# Give the interim evaluation its one GPU before filling the eight-GPU quota.
# An analysis failure must not discard a valid training checkpoint.
eight=$(submit_stage 8gpu --nodes=2 --time=24:00:00 \
  --dependency="afterok:${four},afterany:${interim}")
# CPU analysis needs no GPU, preemption partition, or typed GPU constraint.
export STAGE=interim-analysis
analysis=$(sbatch --parsable --account=torch_pr_500_lzanna --partition=cs --nodes=1 --ntasks=1 \
  --cpus-per-task=16 --mem=128G --time=12:00:00 --kill-on-invalid-dep=yes \
  --dependency="afterok:${interim}" --job-name=scaled-interim-analysis \
  --output="${SCHEDULING_DIR}/interim-analysis-%j.out" --error="${SCHEDULING_DIR}/interim-analysis-%j.err" \
  "${SCHEDULING_DIR}/staged_job.sbatch")
printf 'interim-analysis\t%s\n' "${analysis}" >> "${SCHEDULING_DIR}/jobs.tsv"
export STAGE=interim-viz
viz=$(sbatch --parsable --account=torch_pr_500_lzanna --partition=cs --nodes=1 --ntasks=1 \
  --cpus-per-task=16 --mem=128G --time=08:00:00 --kill-on-invalid-dep=yes \
  --dependency="afterok:${interim}" --job-name=scaled-interim-viz \
  --output="${SCHEDULING_DIR}/interim-viz-%j.out" --error="${SCHEDULING_DIR}/interim-viz-%j.err" \
  "${SCHEDULING_DIR}/staged_job.sbatch")
printf 'interim-viz\t%s\n' "${viz}" >> "${SCHEDULING_DIR}/jobs.tsv"
# Submit a complete replacement chain; some Torch installations reject edits
# to pending jobs through the site job-modification hook.
export TRAIN_NAME="${base_name}" OUTPUT_BASE=/scratch/am16581/runs
export BUNDLE_DIR="${OUTPUT_BASE}/_experiments/${base_name}"
export CODE_COMMIT=acdbcf676867452e2f836dd7d4df3fa6477d76ee
export CODE_LAYER="/scratch/am16581/.apptainer-code-layers/samudra-code-${CODE_COMMIT}.img"
export SIF_PATH=/scratch/am16581/.apptainer-images/physicsnemo-26.05-4f86be45.sif
export IMAGE_REF=ghcr.io/m2lines/ocean-emulator-physicsnemo:26.05-4f86be45a09cb9eba1b51b6273e0189941b02495
export DATA_ROOT=/scratch/am16581/data/om4_onedeg_v3 SCRATCH_DIR=/scratch/am16581
export SIF_DIR=/scratch/am16581/.apptainer-images WANDB_MODE=online
export APPTAINER_CACHEDIR=/scratch/am16581/apptainer-cache
export SINGULARITY_CACHEDIR=/scratch/am16581/singularity-cache
export NCCL_P2P_DISABLE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
submit_final() {
  export PHASE="$1" NAME="${base_name}-$1"
  shift
  local job
  job=$(sbatch --parsable --account=torch_pr_500_lzanna --nodes=1 --ntasks=1 \
    --chdir=/scratch/am16581 --kill-on-invalid-dep=yes --job-name="scaled-final-${PHASE}" \
    --output="${SCHEDULING_DIR}/final-${PHASE}-%j.out" --error="${SCHEDULING_DIR}/final-${PHASE}-%j.err" \
    "$@" "${BUNDLE_DIR}/phase.sbatch")
  printf 'final-%s\t%s\n' "${PHASE}" "${job}" >> "${SCHEDULING_DIR}/jobs.tsv"
  printf '%s' "${job}"
}
final_eval=$(submit_final eval --gres=gpu:1 --constraint=rtx6000 --cpus-per-task=16 --mem=175G \
  --time=04:00:00 --dependency="afterok:${eight}")
submit_final viz --partition=cs --cpus-per-task=32 --mem=256G --time=24:00:00 \
  --dependency="afterok:${final_eval}" > /dev/null
submit_final metrics --partition=cs --cpus-per-task=16 --mem=128G --time=12:00:00 \
  --dependency="afterok:${final_eval}" > /dev/null
submit_final imprint --partition=cs --cpus-per-task=16 --mem=64G --time=12:00:00 \
  --dependency="afterok:${final_eval}" > /dev/null
cat "${SCHEDULING_DIR}/jobs.tsv"
