#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

#SBATCH --job-name=llc-requeue-ledger
#SBATCH --partition=mit_quicktest
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail
spike_root=/orcd/scratch/orcd/008/merose/llc_sharding_spikes_20260903
attempt=${SLURM_RESTART_COUNT:-0}
result_root="$spike_root/results/requeue-ledger-$SLURM_JOB_ID"
"$spike_root/env/bin/python" \
  "$spike_root/scripts/icechunk_requeue_ledger_spike.py" \
  "$result_root" --attempt "$attempt" \
  > "$result_root-attempt-$attempt.json"

if [[ "$attempt" -eq 0 ]]; then
  scontrol requeue "$SLURM_JOB_ID"
fi
