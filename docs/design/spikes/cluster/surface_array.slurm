#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

#SBATCH --job-name=llc-surface
#SBATCH --partition=mit_normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-5%2

set -euo pipefail
export BLOSC_NTHREADS=1
spike_root=/orcd/scratch/orcd/008/merose/llc_sharding_spikes_20260903
inners=(180 240 360 360 180 360)
outers=(1440 1440 1440 4320 720 1440)
time_shards=(2 2 2 2 2 1)
task=${SLURM_ARRAY_TASK_ID:?}
candidate=$("$spike_root/env/bin/python" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["winner"]["candidate_index"])' \
  "$spike_root/results/codec-winner.json")
inner=${inners[$task]}
outer=${outers[$task]}
time_shard=${time_shards[$task]}
"$spike_root/env/bin/python" "$spike_root/scripts/surface_bakeoff.py" \
  "$spike_root/fixtures/cody_face1_t0_t1" \
  "$spike_root/results/surface-$task" \
  --candidate "$candidate" --inner "$inner" --outer "$outer" \
  --time-shard "$time_shard"
