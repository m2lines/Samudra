#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

#SBATCH --job-name=llc-depth
#SBATCH --partition=mit_normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --array=0-8%2

set -euo pipefail
export BLOSC_NTHREADS=1
spike_root=/orcd/scratch/orcd/008/merose/llc_sharding_spikes_20260903
depths=(3 17 51 3 17 51 3 17 51)
inners=(72 72 72 90 90 90 120 120 120)
task=${SLURM_ARRAY_TASK_ID:?}
depth=${depths[$task]}
inner=${inners[$task]}
candidate=$("$spike_root/env/bin/python" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["winner"]["candidate_index"])' \
  "$spike_root/results/codec-winner.json")
"$spike_root/env/bin/python" "$spike_root/scripts/schema_bakeoff.py" \
  "$spike_root/fixtures/cody_face1_t0_t1" \
  "$spike_root/results/depth-$depth-inner-$inner" \
  --candidate "$candidate" --depth-inner "$depth" --inner "$inner" \
  --arrays U Theta --time-count 2
