#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
: "${UTILIZATION_CSV:?Set per-job telemetry path}"
mkdir -p "$(dirname "$UTILIZATION_CSV")"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader,nounits --loop=15 >> "$UTILIZATION_CSV" &
telemetry_pid=$!
trap 'kill "$telemetry_pid" 2>/dev/null || true' EXIT
# A background child plus wait lets the batch shell react immediately to USR1.
# Recovery loads a fully written atomic checkpoint (at most three minutes old).
trap 'scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1
bash "$HOME/slurm_apptainer_train.sbatch" &
harness_pid=$!
wait "$harness_pid"
