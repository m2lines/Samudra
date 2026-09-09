#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

#SBATCH --job-name=llc-layout-v3
#SBATCH --partition=pi_abodner
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --array=0-4%2

set -euo pipefail
export BLOSC_NTHREADS=1
source_root=/orcd/scratch/orcd/008/merose/llc_sharding_spikes_20260903
run_root=${LLC_LAYOUT_RUN_ROOT:?Set LLC_LAYOUT_RUN_ROOT to a new MIT scratch path}
candidates=(A B C D E)
candidate=${candidates[${SLURM_ARRAY_TASK_ID:?}]}
mkdir -p "$run_root/results" "$run_root/logs"
"$source_root/env/bin/python" "$run_root/scripts/layout_tournament.py" \
  "$source_root/fixtures/face1_t0_t7_region720_3600/Theta.npy" \
  "$run_root/results/candidate-$candidate" \
  --candidate "$candidate" \
  > "$run_root/logs/candidate-$candidate-$SLURM_JOB_ID.out" 2>&1
