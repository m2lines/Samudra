#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

#SBATCH --job-name=llc-outer
#SBATCH --partition=mit_normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --array=0-3%2

set -euo pipefail
export BLOSC_NTHREADS=1
spike_root=/orcd/scratch/orcd/008/merose/llc_sharding_spikes_20260903
outers=(720 1440 2160 1440)
time_shards=(1 1 1 2)
task=${SLURM_ARRAY_TASK_ID:?}
outer=${outers[$task]}
time_shard=${time_shards[$task]}
candidate=$("$spike_root/env/bin/python" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["winner"]["candidate_index"])' \
  "$spike_root/results/codec-winner.json")
inner=$("$spike_root/env/bin/python" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["winner"]["inner"])' \
  "$spike_root/results/inner-winner.json")
"$spike_root/env/bin/python" "$spike_root/scripts/schema_bakeoff.py" \
  "$spike_root/fixtures/cody_face1_t0_t1" \
  "$spike_root/results/outer-$outer-time-$time_shard" \
  --candidate "$candidate" --inner "$inner" --outer "$outer" \
  --time-shard "$time_shard" --arrays U Theta --time-count 2
