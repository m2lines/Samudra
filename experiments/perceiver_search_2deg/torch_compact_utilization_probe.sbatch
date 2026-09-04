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
: "${SEARCH_CONFIG:?SEARCH_CONFIG is required relative to the repository root}"

node_root="${TMPDIR:-/tmp}/samudra-utilization-${SLURM_JOB_ID:-local}"
repo="$node_root/repo"
compact_data_root="$node_root/data"
runtime_search="$node_root/search.yaml"
mkdir -p "$repo"

tar -xzf "$CODE_ARCHIVE" -C "$repo"
actual_commit="$(git -C "$repo" rev-parse HEAD)"
if [[ "$actual_commit" != "$CODE_COMMIT" ]]; then
  echo "Expected $CODE_COMMIT from $CODE_ARCHIVE, got $actual_commit" >&2
  exit 1
fi

export PYTHONPATH="$repo/src"

if [[ ! -f "$compact_data_root/_SUCCESS" ]]; then
  test ! -e "$compact_data_root"
  "$PYTHON" - "$SOURCE_DATA_ROOT" "$compact_data_root" <<'PY'
import os
import pathlib
import sys

import dask
import xarray as xr

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
        for base_name in ("uo", "vo", "thetao", "so"):
            level_names = [f"{base_name}_{level}" for level in range(19)]
            if not all(level_name in data for level_name in level_names):
                continue
            levels = data.coords.get("lev", xr.DataArray(range(19))).values
            data[base_name] = xr.concat(
                [data[level_name] for level_name in level_names], dim="lev"
            ).assign_coords(lev=("lev", levels))
            data = data.drop_vars(level_names)
        if name == "OM4":
            data = data.chunk({"time": 1, "lev": 19})
        data.to_zarr(os.path.join(destination_root, f"{name}.zarr"))

pathlib.Path(destination_root, "_SUCCESS").touch()
PY
fi

cd "$repo"
sed "s|__COMPACT_DATA_ROOT__|$compact_data_root|g" \
  "$repo/$SEARCH_CONFIG" > "$runtime_search"
exec "$PYTHON" -m samudra.search "$runtime_search"
