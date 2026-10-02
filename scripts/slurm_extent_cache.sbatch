#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
# Submit through Alpha/Grace, after the verified quarter-degree transfer succeeds.
set -euo pipefail
root=$1
code=$2
export SAMUDRA_CODE_COMMIT=$3
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
python=/mnt/home/jrusak/data/obs_full_range/envs/grace-arm64/bin/python
"$python" "$code/scripts/prepare_om4_patch_cache.py" \
  --source /projects/ny/lz1955/multiscale/jrusak/data/om4/v2026-09/om4_quarterdeg \
  --transfer-success /projects/ny/lz1955/multiscale/jrusak/transfers/2026-10-02-om4-quarterdeg-averaged/reports/114235/SUCCESS.json \
  --grid /mnt/home/jrusak/data/obs_full_range/d-observation-pilot/samples/grid.npz \
  --output /projects/ny/lz1955/multiscale/jrusak/data/om4/v2026-09/om4_quarterdeg_patch_cache \
  --workers 8
