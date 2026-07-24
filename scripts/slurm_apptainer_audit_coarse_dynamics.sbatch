#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

# Run the synchronized latent-dynamics audit against one completed S2 run.
# Slurm resources and preemption routing are supplied at submission time.

set -euo pipefail

for variable_name in \
  RUN_DIR \
  INVERSE_CHECKPOINT \
  DATA_ROOT \
  SIF_PATH \
  CODE_LAYER \
  AUDIT_SCRIPT \
  INVERSE_AUDIT_SCRIPT; do
  if [[ -z "${!variable_name:-}" ]]; then
    echo "Missing required environment variable: ${variable_name}" >&2
    exit 2
  fi
done

CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/saved_nets/ckpt.pt}"
CONFIG="${CONFIG:-configs/samudra_multi_om4/validate_cross_1_halfdeg_coarse_latent_dynamics.yaml}"
MAX_BATCHES="${MAX_BATCHES:-32}"

for required_file in \
  "${CHECKPOINT}" \
  "${INVERSE_CHECKPOINT}" \
  "${SIF_PATH}" \
  "${CODE_LAYER}" \
  "${CODE_LAYER}.sha256" \
  "${CODE_LAYER}.json" \
  "${AUDIT_SCRIPT}" \
  "${INVERSE_AUDIT_SCRIPT}"; do
  if [[ ! -s "${required_file}" ]]; then
    echo "Required audit input is missing or empty: ${required_file}" >&2
    exit 3
  fi
done
if [[ ! "${MAX_BATCHES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_BATCHES must be a positive integer." >&2
  exit 2
fi

if command -v apptainer >/dev/null 2>&1; then
  APPTAINER_BIN=apptainer
elif command -v singularity >/dev/null 2>&1; then
  APPTAINER_BIN=singularity
else
  echo "Neither apptainer nor singularity is available." >&2
  exit 4
fi

(
  cd "$(dirname "${CODE_LAYER}")"
  sha256sum --check --status "$(basename "${CODE_LAYER}.sha256")"
)

export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/tmp}"
export APPTAINERENV_PYTHONPATH=/opt/samudra-code/src:/tmp

"${APPTAINER_BIN}" exec \
  --nv \
  --overlay "${CODE_LAYER}:ro" \
  --bind "${DATA_ROOT}:${DATA_ROOT}:ro" \
  --bind "${RUN_DIR}:${RUN_DIR}:ro" \
  --bind "$(dirname "${INVERSE_CHECKPOINT}"):$(dirname "${INVERSE_CHECKPOINT}"):ro" \
  --bind "${AUDIT_SCRIPT}:/tmp/audit_coarse_dynamics.py:ro" \
  --bind "${INVERSE_AUDIT_SCRIPT}:/tmp/scripts/audit_coarse_inverse.py:ro" \
  --pwd /opt/samudra-code \
  "${SIF_PATH}" \
  /workspace/.venv/bin/python \
  /tmp/audit_coarse_dynamics.py \
  "${CONFIG}" \
  "${CHECKPOINT}" \
  --inverse-checkpoint "${INVERSE_CHECKPOINT}" \
  --data-root "${DATA_ROOT}" \
  --backend cuda \
  --max-batches "${MAX_BATCHES}" \
  | tee "${RUN_DIR}/coarse_dynamics_audit.log"
