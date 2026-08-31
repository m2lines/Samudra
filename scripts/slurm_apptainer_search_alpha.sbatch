#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

# Single-GPU Alpha worker for samudra.search. Resource selection is supplied by
# the search executor; this file only establishes the immutable runtime.

set -euo pipefail

: "${CODE_DIR:?CODE_DIR must name a clean committed checkout}"
: "${CODE_COMMIT:?CODE_COMMIT must name its full commit}"
: "${SIF_PATH:?SIF_PATH must name the experiment container}"
: "${SAMUDRA_MODULE:?SAMUDRA_MODULE is required}"
: "${SAMUDRA_MODULE_ARGS:?SAMUDRA_MODULE_ARGS is required}"
: "${SCRATCH_DIR:?SCRATCH_DIR is required}"

test "$(git -C "$CODE_DIR" rev-parse HEAD)" = "$CODE_COMMIT"
test -z "$(git -C "$CODE_DIR" status --porcelain)"
test -s "$SIF_PATH"

source /etc/profile.d/modules.sh
module load apptainer

export APPTAINERENV_PYTHONPATH="$SCRATCH_DIR/python-site:/opt/samudra-code/src"
export APPTAINERENV_WANDB_MODE="${WANDB_MODE:-online}"
export APPTAINERENV_WANDB_API_KEY="${WANDB_API_KEY:-}"
export APPTAINERENV_SAMUDRA_CODE_COMMIT="$CODE_COMMIT"
export APPTAINERENV_WANDB_GIT_COMMIT="$CODE_COMMIT"
export APPTAINERENV_WANDB_GIT_REMOTE_URL="$(git -C "$CODE_DIR" config --get remote.origin.url)"

read -r -a module_args <<<"$SAMUDRA_MODULE_ARGS"
srun --ntasks=1 apptainer exec --nv \
  --bind "$CODE_DIR:/opt/samudra-code:ro" \
  --bind "$SCRATCH_DIR:$SCRATCH_DIR" \
  --pwd /opt/samudra-code \
  "$SIF_PATH" \
  /workspace/.venv/bin/python -m "$SAMUDRA_MODULE" "${module_args[@]}"
