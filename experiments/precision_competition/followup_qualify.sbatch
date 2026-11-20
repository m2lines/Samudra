#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=precision-ema-qualify
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --requeue

set -euo pipefail
: "${RUN_ROOT:?}" "${SOURCE_COMMIT:?}" "${SIF_PATH:?}" "${MANIFEST_SHA256:?}"
cd "$RUN_ROOT/code"
echo "$MANIFEST_SHA256  experiments/precision_competition/followup-qualification.json" | sha256sum -c -
export APPTAINER_CACHEDIR=/scratch/jr7309/apptainer-cache
export APPTAINER_TMPDIR="${SLURM_TMPDIR:-/tmp}"
export PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv
apptainer exec --nv --bind "$RUN_ROOT:$RUN_ROOT" --pwd "$RUN_ROOT/code" "$SIF_PATH" \
    /usr/bin/env TORCH_CUDNN_V8_API_DISABLED=1 PYTHONUNBUFFERED=1 \
    /workspace/.venv/bin/python -m experiments.precision_competition.followup_qualify --output "$RUN_ROOT/smoke" --scales experiments/precision_competition/followup-qualification.scales.json
