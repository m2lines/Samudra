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
  --bind /opt/slurm:/opt/slurm:ro \
  --bind /usr/lib64/libmunge.so.2:/usr/lib64/libmunge.so.2:ro \
  --bind /usr/lib64/libmunge.so.2.0.0:/usr/lib64/libmunge.so.2.0.0:ro \
  --bind /run/munge:/run/munge:ro \
  --bind /etc/passwd:/etc/passwd:ro \
  --bind /etc/group:/etc/group:ro \
  --pwd /opt/samudra-code \
  --env PATH=/opt/slurm/bin:/workspace/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin \
  --env LD_LIBRARY_PATH=/opt/slurm/lib64/slurm:/usr/lib64 \
  --env PYTHONPATH=/opt/samudra-code/src \
  --env "SAMUDRA_CODE_COMMIT=${SAMUDRA_CODE_COMMIT}" \
  "${SIF_PATH}" \
  /workspace/.venv/bin/python "$@"
