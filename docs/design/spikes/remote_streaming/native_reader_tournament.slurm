#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

#SBATCH --job-name=llc-native-read
#SBATCH --partition=pi_abodner
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-9%2

set -euo pipefail
export BLOSC_NTHREADS=1
run_root=${LLC_LAYOUT_RUN_ROOT:?Set LLC_LAYOUT_RUN_ROOT to the MIT scratch path}
source_root=/orcd/scratch/orcd/008/merose/llc_sharding_spikes_20260903
candidates=(A B C D E)
readers=(zarrs tensorstore)
task=${SLURM_ARRAY_TASK_ID:?}
candidate=${candidates[$((task % 5))]}
reader=${readers[$((task / 5))]}
"$run_root/native-env/bin/python" "$run_root/scripts/native_reader_tournament.py" \
  "$source_root/fixtures/face1_t0_t7_region720_3600/Theta.npy" \
  "$run_root/results/candidate-$candidate/candidate.zarr" \
  "$run_root/results/candidate-$candidate/native-$reader.json" \
  --reader "$reader" \
  > "$run_root/logs/native-$reader-$candidate-$SLURM_JOB_ID.out" 2>&1
