#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
: "${UTILIZATION_CSV:?Set the per-job telemetry path}"
mkdir -p "$(dirname "$UTILIZATION_CSV")"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader,nounits --loop=5 > "$UTILIZATION_CSV" &
telemetry_pid=$!
trap 'kill "$telemetry_pid" 2>/dev/null || true' EXIT
bash "$HOME/slurm_apptainer_train.sbatch"
