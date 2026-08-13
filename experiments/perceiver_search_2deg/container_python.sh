#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

# Execute search-controller Python in the same immutable environment as the
# training workers. SIF_PATH and CODE_LAYER are inherited through Slurm.

set -euo pipefail

: "${SIF_PATH:?SIF_PATH must identify the experiment container}"
: "${CODE_LAYER:?CODE_LAYER must identify the experiment code layer}"

exec apptainer exec \
  --overlay "${CODE_LAYER}:ro" \
  --pwd /opt/samudra-code \
  --env PYTHONPATH=/opt/samudra-code/src \
  "${SIF_PATH}" \
  /workspace/.venv/bin/python "$@"
