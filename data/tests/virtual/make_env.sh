#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

# Build the environment that tests ocean_preprocessing/virtualize_om4.py, from the
# script's own PEP 723 metadata (VirtualiZarr needs zarr-python 3; the rest of
# data/ uses 2).
#
# Usage: data/tests/virtual/make_env.sh <venv-dir> [pinned|canary]
#   pinned  exactly the packages in virtualize_om4.py.lock (the default)
#   canary  the newest versions the script's header allows
# Writes <venv-dir>/requirements.txt, the resolved packages, for the record.
set -euo pipefail

venv=$1
lane=${2:-pinned}
root=$(cd "$(dirname "$0")/../../.." && pwd)
script=$root/data/ocean_preprocessing/virtualize_om4.py

uv venv --quiet --python 3.12 "$venv"
case $lane in
  pinned)
    uv export --quiet --script "$script" --locked --no-hashes >"$venv/requirements.txt"
    tools=(pytest==9.2.0 hypothesis==6.170.13 pytest-reportlog==1.0.0 h5netcdf==1.7.3 cftime)
    ;;
  canary)
    # A copy without the lock next to it resolves afresh.
    fresh=$(mktemp -d)
    cp "$script" "$fresh/"
    uv export --quiet --script "$fresh/virtualize_om4.py" --no-hashes >"$venv/requirements.txt"
    tools=(pytest hypothesis pytest-reportlog h5netcdf cftime)
    ;;
  *)
    echo "unknown lane: $lane (expected pinned or canary)" >&2
    exit 2
    ;;
esac
# Test tools; h5netcdf and cftime write the NetCDF fixtures.
uv pip install --quiet --python "$venv" -r "$venv/requirements.txt" "${tools[@]}"
# The tests import ocean_preprocessing.virtualize_om4 and the property tooling.
uv pip install --quiet --python "$venv" --no-deps -e "$root/verification" -e "$root/data"
