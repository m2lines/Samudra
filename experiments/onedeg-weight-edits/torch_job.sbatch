#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

# Submit with PHASE=eval or PHASE=viz after the pinned code layer is built.
# Resource requests are supplied by sbatch, not inferred by this script.
set -euo pipefail
: "${BUNDLE_DIR:?}" "${SIF_PATH:?}" "${CODE_LAYER:?}" "${NAME:?}" "${PHASE:?}"
: "${OUTPUT_BASE:?}" "${DATA_ROOT:?}" "${WANDB_API_KEY:?Online W&B credentials required}"
export WANDB_MODE=online
export WANDB_RUN_GROUP=onedeg-weight-edit-20261103
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
test ! -e "${OUTPUT_BASE}/${NAME}"
(cd "${BUNDLE_DIR}" && sha256sum --check --status bundle.sha256)
(cd "$(dirname "${CODE_LAYER}")" && sha256sum --check --status "$(basename "${CODE_LAYER}.sha256")")
export WANDB_GIT_COMMIT
WANDB_GIT_COMMIT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["code_commit"])' "${CODE_LAYER}.json")"
export SAMUDRA_CODE_COMMIT="${WANDB_GIT_COMMIT}"
export SAMUDRA_CODE_REPO_URL=https://github.com/m2lines/Samudra.git
export SAMUDRA_CODE_LAYER_SHA256
SAMUDRA_CODE_LAYER_SHA256="$(awk 'NR==1 {print $1}' "${CODE_LAYER}.sha256")"
export SAMUDRA_CONTAINER_SIF_PATH="${SIF_PATH}"
export SAMUDRA_CONTAINER_IMAGE_REF=ghcr.io/m2lines/ocean-emulator-physicsnemo:26.05-4f86be45a09cb9eba1b51b6273e0189941b02495
export SAMUDRA_CONTAINER_GIT_COMMIT=4f86be45a09cb9eba1b51b6273e0189941b02495
apptainer exec --overlay "${CODE_LAYER}:ro" "${SIF_PATH}" \
  bash -c 'cmp -s /opt/samudra-code/uv.lock /workspace/uv.lock && cmp -s /opt/samudra-code/pyproject.toml /workspace/pyproject.toml'

GPU_ARGS=()
if [[ "${PHASE}" == eval ]]; then
  : "${CKPT_PATH:?}"
  (cd "${BUNDLE_DIR}/checkpoints" && sha256sum --check --status checkpoints.sha256)
  GPU_ARGS=(--nv)
  RUN_ARGS=("${BUNDLE_DIR}/onedeg_tracked_eval.py" "${BUNDLE_DIR}/torch_eval.yaml"
    "--ckpt_path=${CKPT_PATH}" "--experiment.name=${NAME}"
    "--experiment.data_root=${DATA_ROOT}" "--experiment.base_output_dir=${OUTPUT_BASE}")
elif [[ "${PHASE}" == viz ]]; then
  RUN_ARGS=("${BUNDLE_DIR}/onedeg_tracked_viz.py" "${BUNDLE_DIR}/torch_viz.yaml"
    "--name=${NAME}" "--data_root=${DATA_ROOT}" "--base_output_dir=${OUTPUT_BASE}")
else
  echo "Unknown PHASE: ${PHASE}" >&2
  exit 2
fi
export APPTAINERENV_PYTHONPATH=/opt/samudra-code/src
export APPTAINERENV_WANDB_GIT_COMMIT="${WANDB_GIT_COMMIT}"
export APPTAINERENV_WANDB_API_KEY="${WANDB_API_KEY}"
export APPTAINERENV_WANDB_MODE=online
export APPTAINERENV_WANDB_RUN_GROUP="${WANDB_RUN_GROUP}"
srun --ntasks=1 apptainer exec "${GPU_ARGS[@]}" --overlay "${CODE_LAYER}:ro" \
  --bind "${BUNDLE_DIR}:${BUNDLE_DIR}:ro,${DATA_ROOT}:${DATA_ROOT}:ro,${OUTPUT_BASE}:${OUTPUT_BASE}" \
  --pwd "${OUTPUT_BASE}" "${SIF_PATH}" \
  /workspace/.venv/bin/python "${RUN_ARGS[@]}"
cp "${CODE_LAYER}.json" "${OUTPUT_BASE}/${NAME}/source-manifest.json"
cp "${BUNDLE_DIR}/bundle.sha256" "${OUTPUT_BASE}/${NAME}/bundle.sha256"
