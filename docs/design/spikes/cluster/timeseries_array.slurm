#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

#SBATCH --job-name=llc-time-layout
#SBATCH --partition=pi_abodner
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-3%2

set -euo pipefail
export BLOSC_NTHREADS=1
spike_root=/orcd/scratch/orcd/008/merose/llc_sharding_spikes_20260903
time_shards=(1 2 4 8)
time_shard=${time_shards[${SLURM_ARRAY_TASK_ID:?}]}
candidate=$("$spike_root/env/bin/python" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["winner"]["candidate_index"])' \
  "$spike_root/results/codec-winner.json")
"$spike_root/env/bin/python" "$spike_root/scripts/timeseries_bakeoff.py" \
  "$spike_root/fixtures/face1_t0_t23_tile" \
  "$spike_root/results/time-$time_shard" \
  --candidate "$candidate" --time-shard "$time_shard"
