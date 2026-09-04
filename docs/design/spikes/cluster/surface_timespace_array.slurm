#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

#SBATCH --job-name=llc-surface-timespace
#SBATCH --partition=pi_abodner
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-4%2

set -euo pipefail
export BLOSC_NTHREADS=1
spike_root=/orcd/scratch/orcd/008/merose/llc_sharding_spikes_20260903
time_shards=(2 8 24 8 24)
outers=(4320 2160 1080 1440 720)
task=${SLURM_ARRAY_TASK_ID:?}
time_shard=${time_shards[$task]}
outer=${outers[$task]}
candidate=$("$spike_root/env/bin/python" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["winner"]["candidate_index"])' \
  "$spike_root/results/codec-winner.json")
"$spike_root/env/bin/python" "$spike_root/scripts/surface_timespace_bakeoff.py" \
  "$spike_root/fixtures/face1_eta_t0_t23_full" \
  "$spike_root/results/surface-timespace-$task" \
  --candidate "$candidate" --time-shard "$time_shard" --outer "$outer"
