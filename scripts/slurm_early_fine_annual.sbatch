#!/bin/bash -l
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
root=$1
code=$2
stage=$3
export SAMUDRA_CODE_COMMIT=$4
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 WANDB_MODE=disabled
export APPTAINER_CACHEDIR=/scratch/jr7309/apptainer-cache
export APPTAINER_TMPDIR=${SLURM_TMPDIR:-/tmp}
image=/scratch/jr7309/.apptainer-images/ghcr.io-m2lines-ocean-emulator-physicsnemo-26.05-9cd36b1bdcf4921fb027494afa4157e217818ccd.sif
if ! command -v apptainer >/dev/null; then
  module load apptainer/1.5.2
fi
flags=()
args=()
if [[ "$stage" == evaluate ]]; then
  flags+=(--nv)
  args+=(--index "$SLURM_ARRAY_TASK_ID")
fi
srun apptainer exec "${flags[@]}" --bind "/scratch:/scratch:ro,$root:$root:rw,$root/sources:$root/sources:ro,$code:/pilot-code:ro" "$image" \
 env PYTHONPATH="/pilot-code/src:/pilot-code:$root/runtime" \
 /workspace/.venv/bin/python /pilot-code/scripts/run_early_fine_annual.py \
 --root "$root" --stage "$stage" "${args[@]}"
