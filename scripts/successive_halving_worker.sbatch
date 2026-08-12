#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

# Resource directives are supplied by successive_halving.py.
set -euo pipefail

: "${HALVING_PYTHON:?Missing HALVING_PYTHON}"
: "${HALVING_SCRIPT:?Missing HALVING_SCRIPT}"
: "${HALVING_MANIFEST:?Missing HALVING_MANIFEST}"
: "${HALVING_STATE:?Missing HALVING_STATE}"
: "${HALVING_RUNG:?Missing HALVING_RUNG}"
: "${SLURM_ARRAY_TASK_ID:?This worker must run as a Slurm array task}"

anchor_args=()
if [[ "${HALVING_ANCHOR:-0}" == "1" ]]; then
  anchor_args+=(--anchor)
fi

exec "${HALVING_PYTHON}" "${HALVING_SCRIPT}" run-task \
  "${HALVING_MANIFEST}" \
  "${HALVING_STATE}" \
  "${HALVING_RUNG}" \
  "${SLURM_ARRAY_TASK_ID}" \
  "${anchor_args[@]}"
