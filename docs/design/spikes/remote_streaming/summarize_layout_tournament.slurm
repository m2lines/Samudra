#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

#SBATCH --job-name=llc-layout-summary
#SBATCH --partition=mit_quicktest
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail
source_root=/orcd/scratch/orcd/008/merose/llc_sharding_spikes_20260903
run_root=${LLC_LAYOUT_RUN_ROOT:?Set LLC_LAYOUT_RUN_ROOT to the MIT scratch path}
"$source_root/env/bin/python" "$run_root/scripts/summarize_layout_tournament.py" \
  "$run_root/results" "$run_root/results/summary.json" \
  > "$run_root/logs/summary-$SLURM_JOB_ID.out" 2>&1
