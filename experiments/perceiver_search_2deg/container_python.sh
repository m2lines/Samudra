#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

# Execute search-controller Python in the same immutable environment as the
# training workers. SIF_PATH and CODE_LAYER are inherited through Slurm.

set -euo pipefail

: "${SIF_PATH:?SIF_PATH must identify the experiment container}"
: "${CODE_LAYER:?CODE_LAYER must identify the experiment code layer}"

CODE_LAYER_MANIFEST="${CODE_LAYER}.json"
if [[ ! -s "${CODE_LAYER_MANIFEST}" ]]; then
  echo "Missing code-layer manifest: ${CODE_LAYER_MANIFEST}" >&2
  exit 3
fi
SAMUDRA_CODE_COMMIT="$(
  python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["code_commit"])' \
    "${CODE_LAYER_MANIFEST}"
)"

exec apptainer exec \
  --overlay "${CODE_LAYER}:ro" \
  --pwd /opt/samudra-code \
  --env PYTHONPATH=/opt/samudra-code/src \
  --env "SAMUDRA_CODE_COMMIT=${SAMUDRA_CODE_COMMIT}" \
  "${SIF_PATH}" \
  /workspace/.venv/bin/python "$@"
