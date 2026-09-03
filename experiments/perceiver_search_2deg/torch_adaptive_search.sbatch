#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

# Run the allocation-aware search controller directly in the batch allocation.
# It launches exclusive srun worker steps and therefore must not itself run
# beneath srun.

set -euo pipefail

: "${SAMUDRA_REPO:?SAMUDRA_REPO is required}"
: "${SEARCH_CONFIG:?SEARCH_CONFIG is required}"

cd "$SAMUDRA_REPO"
test -x .venv/bin/python
test -f "$SEARCH_CONFIG"

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

exec .venv/bin/python -m samudra.search "$SEARCH_CONFIG"
