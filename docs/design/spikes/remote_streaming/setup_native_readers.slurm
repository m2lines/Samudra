#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

#SBATCH --job-name=llc-native-env
#SBATCH --partition=mit_quicktest
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
run_root=${LLC_LAYOUT_RUN_ROOT:?Set LLC_LAYOUT_RUN_ROOT to the MIT scratch path}
python=/usr/bin/python3.12
env_path="$run_root/native-env"
"$python" -m venv "$env_path"
"$env_path/bin/python" -m pip install --upgrade pip
"$env_path/bin/python" -m pip install \
  'numpy==2.5.2' \
  'psutil==7.2.2' \
  'tensorstore==0.1.85' \
  'zarr==3.3.0' \
  'zarrs==0.2.3'
"$env_path/bin/python" - <<'PY'
import importlib.metadata
import json

print(
    json.dumps(
        {
            name: importlib.metadata.version(name)
            for name in ("tensorstore", "zarr", "zarrs")
        },
        indent=2,
    )
)
PY
