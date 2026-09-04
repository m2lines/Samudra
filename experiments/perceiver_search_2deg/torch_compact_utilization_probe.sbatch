#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

# Build the compact smoke slice once, then measure training in the same
# allocation. The immutable source checkout lives on node-local storage so a
# congested shared filesystem cannot stall Git/Python source metadata reads.

set -euo pipefail

: "${CODE_COMMIT:?CODE_COMMIT is required}"
: "${CODE_ARCHIVE:?CODE_ARCHIVE is required}"
: "${PYTHON:?PYTHON is required}"
: "${SOURCE_DATA_ROOT:?SOURCE_DATA_ROOT is required}"
: "${COMPACT_DATA_ROOT:?COMPACT_DATA_ROOT is required}"
: "${SEARCH_CONFIG:?SEARCH_CONFIG is required relative to the repository root}"

node_root="${TMPDIR:-/tmp}/samudra-utilization-${SLURM_JOB_ID:-local}"
repo="$node_root/repo"
mkdir -p "$repo"

tar -xzf "$CODE_ARCHIVE" -C "$repo"
actual_commit="$(git -C "$repo" rev-parse HEAD)"
if [[ "$actual_commit" != "$CODE_COMMIT" ]]; then
  echo "Expected $CODE_COMMIT from $CODE_ARCHIVE, got $actual_commit" >&2
  exit 1
fi

export PYTHONPATH="$repo/src"

if [[ ! -f "$COMPACT_DATA_ROOT/OM4.zarr/.zmetadata" ]]; then
  test ! -e "$COMPACT_DATA_ROOT"
  "$PYTHON" - "$SOURCE_DATA_ROOT" "$COMPACT_DATA_ROOT" <<'PY'
import os
import pathlib
import sys

import dask
import xarray as xr

from samudra.utils.data import compact_dataset

source_root, destination_root = sys.argv[1:]
pathlib.Path(destination_root).mkdir(parents=True)
workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))

with dask.config.set(scheduler="threads", num_workers=workers):
    for name in ("OM4", "OM4_means", "OM4_stds"):
        chunks = {"time": 100} if name == "OM4" else {}
        data = xr.open_dataset(
            os.path.join(source_root, f"{name}.zarr"),
            engine="zarr",
            chunks=chunks,
        )
        if name == "OM4":
            data = data.isel(time=slice(1240, 1830))
        data = compact_dataset(data)
        if name == "OM4":
            data = data.chunk({"time": 1, "lev": 19})
        data.to_zarr(os.path.join(destination_root, f"{name}.zarr"))
PY
fi

cd "$repo"
exec "$PYTHON" -m samudra.search "$repo/$SEARCH_CONFIG"
