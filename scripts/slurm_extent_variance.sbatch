#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
code=$1
export SAMUDRA_CODE_COMMIT=$2
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$code:$code/src:/projects/ny/lz1955/multiscale/jrusak/runs/2026-10-02-extent-om4/cpu-runtime"
python=/mnt/home/jrusak/data/obs_full_range/envs/grace-arm64/bin/python
"$python" "$code/scripts/prepare_extent_variance.py" \
  --source /projects/ny/lz1955/multiscale/jrusak/data/om4/v2026-09/om4_quarterdeg \
  --global-om4 /mnt/home/jrusak/data/diffusion-interior-beta/data/om4_onedeg_v3 \
  --transfer-success /projects/ny/lz1955/multiscale/jrusak/transfers/2026-10-02-om4-quarterdeg-averaged/reports/114235/SUCCESS.json \
  --output /projects/ny/lz1955/multiscale/jrusak/data/om4/v2026-09/om4_quarterdeg_surface_variance_1deg \
  --workers 8
