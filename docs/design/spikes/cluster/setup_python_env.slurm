#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

#SBATCH --job-name=llc-spike-env
#SBATCH --partition=mit_quicktest
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

spike_root=/orcd/scratch/orcd/008/merose/llc_sharding_spikes_20260903
env_path="$spike_root/env"

mkdir -p "$spike_root/logs"
/usr/bin/python3.12 -m venv "$env_path"
"$env_path/bin/python" -m pip install --upgrade pip
"$env_path/bin/python" -m pip install \
  'dask[array]>=2025.1,<2027' \
  'icechunk==2.2.0' \
  'numpy>=2.1,<3' \
  'psutil>=6,<8' \
  'xarray>=2025.1,<2027' \
  'zarr==3.3.0'

"$env_path/bin/python" - <<'PY'
import importlib
import json

names = ["dask", "icechunk", "numpy", "psutil", "xarray", "zarr"]
print(json.dumps({name: importlib.import_module(name).__version__ for name in names}, indent=2))
PY
