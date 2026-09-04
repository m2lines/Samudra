#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

# Build the compact smoke slice once, then measure training in the same
# allocation. The immutable source checkout lives on node-local storage so a
# congested shared filesystem cannot stall Git/Python source metadata reads.

set -euo pipefail

: "${CODE_REF:?CODE_REF is required}"
: "${CODE_COMMIT:?CODE_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${SOURCE_DATA_ROOT:?SOURCE_DATA_ROOT is required}"
: "${COMPACT_DATA_ROOT:?COMPACT_DATA_ROOT is required}"
: "${SEARCH_CONFIG:?SEARCH_CONFIG is required relative to the repository root}"

node_root="${TMPDIR:-/tmp}/samudra-utilization-${SLURM_JOB_ID:-local}"
repo="$node_root/repo"
mkdir -p "$node_root"

git clone --depth 1 --single-branch --branch "$CODE_REF" \
  https://github.com/m2lines/Samudra.git "$repo"
actual_commit="$(git -C "$repo" rev-parse HEAD)"
if [[ "$actual_commit" != "$CODE_COMMIT" ]]; then
  echo "Expected $CODE_COMMIT from $CODE_REF, got $actual_commit" >&2
  exit 1
fi

export PYTHONPATH="$repo/src"

if [[ ! -f "$COMPACT_DATA_ROOT/OM4.zarr/.zmetadata" ]]; then
  test ! -e "$COMPACT_DATA_ROOT"
  "$PYTHON" "$repo/scripts/clone_data.py" "$COMPACT_DATA_ROOT" \
    --source "$SOURCE_DATA_ROOT" \
    --time_start 1240 \
    --time_end 1830 \
    --write_time_chunks 1 \
    --compact_variables \
    --local_cluster
fi

cd "$repo"
exec "$PYTHON" -m samudra.search "$repo/$SEARCH_CONFIG"
