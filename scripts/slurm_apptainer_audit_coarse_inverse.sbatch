#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

# Run the coarse learned-inverse audit against one completed training directory.
# Slurm resources and preemption routing are supplied at submission time.

set -euo pipefail

for variable_name in RUN_DIR DATA_ROOT SIF_PATH CODE_LAYER AUDIT_SCRIPT; do
  if [[ -z "${!variable_name:-}" ]]; then
    echo "Missing required environment variable: ${variable_name}" >&2
    exit 2
  fi
done

for required_file in \
  "${RUN_DIR}/config.yaml" \
  "${RUN_DIR}/saved_nets/ckpt.pt" \
  "${SIF_PATH}" \
  "${CODE_LAYER}" \
  "${CODE_LAYER}.sha256" \
  "${CODE_LAYER}.json" \
  "${AUDIT_SCRIPT}"; do
  if [[ ! -s "${required_file}" ]]; then
    echo "Required audit input is missing or empty: ${required_file}" >&2
    exit 3
  fi
done

if command -v apptainer >/dev/null 2>&1; then
  APPTAINER_BIN=apptainer
elif command -v singularity >/dev/null 2>&1; then
  APPTAINER_BIN=singularity
else
  echo "Neither apptainer nor singularity is available." >&2
  exit 4
fi

AGREEMENT_BATCHES="${AGREEMENT_BATCHES:-32}"
if [[ ! "${AGREEMENT_BATCHES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "AGREEMENT_BATCHES must be a positive integer." >&2
  exit 2
fi

(
  cd "$(dirname "${CODE_LAYER}")"
  sha256sum --check --status "$(basename "${CODE_LAYER}.sha256")"
)

export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/tmp}"
export APPTAINERENV_PYTHONPATH=/opt/samudra-code/src

"${APPTAINER_BIN}" exec \
  --nv \
  --overlay "${CODE_LAYER}:ro" \
  --bind "${DATA_ROOT}:${DATA_ROOT}:ro" \
  --bind "${RUN_DIR}:${RUN_DIR}" \
  --bind "${AUDIT_SCRIPT}:/tmp/audit_coarse_inverse.py:ro" \
  --pwd /opt/samudra-code \
  "${SIF_PATH}" \
  /workspace/.venv/bin/python \
  /tmp/audit_coarse_inverse.py \
  "${RUN_DIR}/config.yaml" \
  "${RUN_DIR}/saved_nets/ckpt.pt" \
  --data-root "${DATA_ROOT}" \
  --backend cuda \
  --agreement-batches "${AGREEMENT_BATCHES}" \
  | tee "${RUN_DIR}/coarse_inverse_audit.log"
