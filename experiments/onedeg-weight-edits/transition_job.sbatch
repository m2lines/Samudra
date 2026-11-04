#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
: "${BUNDLE_DIR:?}" "${SIF_PATH:?}" "${CODE_LAYER:?}" "${RUN_OUTPUT:?}"
: "${DATA_ROOT:?}" "${CHECKPOINT_DIR:?}" "${WANDB_API_KEY:?Online tracking required}"
test ! -e "${RUN_OUTPUT}"
(cd "${BUNDLE_DIR}" && sha256sum --check --status bundle.sha256)
(cd "${CHECKPOINT_DIR}" && sha256sum --check --status checkpoints.sha256)
(cd "$(dirname "${CODE_LAYER}")" && sha256sum --check --status "$(basename "${CODE_LAYER}.sha256")")
export WANDB_MODE=online PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export WANDB_GIT_COMMIT
WANDB_GIT_COMMIT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["code_commit"])' "${CODE_LAYER}.json")"
export SAMUDRA_CODE_LAYER_SHA256
SAMUDRA_CODE_LAYER_SHA256="$(awk 'NR==1 {print $1}' "${CODE_LAYER}.sha256")"
export APPTAINERENV_PYTHONPATH="${BUNDLE_DIR}:/opt/samudra-code/src"
export APPTAINERENV_WANDB_API_KEY="${WANDB_API_KEY}"
export APPTAINERENV_WANDB_MODE=online
export APPTAINERENV_WANDB_GIT_COMMIT="${WANDB_GIT_COMMIT}"
apptainer exec --overlay "${CODE_LAYER}:ro" "${SIF_PATH}" \
  bash -c 'cmp -s /opt/samudra-code/uv.lock /workspace/uv.lock && cmp -s /opt/samudra-code/pyproject.toml /workspace/pyproject.toml'
srun --ntasks=1 apptainer exec --nv --overlay "${CODE_LAYER}:ro" \
  --bind "${BUNDLE_DIR}:${BUNDLE_DIR}:ro,${DATA_ROOT}:${DATA_ROOT}:ro,${CHECKPOINT_DIR}:${CHECKPOINT_DIR}:ro,$(dirname "${RUN_OUTPUT}"):$(dirname "${RUN_OUTPUT}")" \
  --pwd /opt/samudra-code "${SIF_PATH}" \
  /workspace/.venv/bin/python -m scripts.onedeg_transition_benchmark \
    --inputs "${BUNDLE_DIR}/inputs" --data-root "${DATA_ROOT}" \
    --checkpoint "${CHECKPOINT_DIR}/samudra2_onedeg_ema_ckpt.pt" \
    --edited-checkpoint "${CHECKPOINT_DIR}/samudra2_onedeg_ema_carry_balanced_l15_0.5.pt" \
    --output "${RUN_OUTPUT}"
cp "${CODE_LAYER}.json" "${RUN_OUTPUT}/source-manifest.json"
cp "${BUNDLE_DIR}/bundle.sha256" "${RUN_OUTPUT}/bundle.sha256"
