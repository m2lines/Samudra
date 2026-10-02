#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
# Resource requests and dependencies are provided by the recorded submission.
set -euo pipefail
stage=$1
arm=$2
root=$3
code=$4
export SAMUDRA_CODE_COMMIT=$5
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
image=ghcr.io/m2lines/ocean-emulator-physicsnemo@sha256:d7cf627257df3e1fda74202435e00f55c15ebe2d2be9648f47daf31905f51bad
deps=/mnt/home/jrusak/data/obs_full_range/d-observation-pilot/code/deps
srun --nodes=1 --ntasks=1 --container-image="$image" --container-mount-home \
  --container-mounts="$code:/extent-code,/projects/ny/lz1955/multiscale:/projects/ny/lz1955/multiscale" \
  env PYTHONPATH="/extent-code/src:$deps" \
  bash /extent-code/scripts/extent_container.sh "$stage" "$arm" "$root"
