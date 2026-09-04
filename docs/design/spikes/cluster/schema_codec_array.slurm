#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

#SBATCH --job-name=llc-codecs
#SBATCH --partition=mit_normal
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --array=0-6%2

set -euo pipefail
export BLOSC_NTHREADS=1
spike_root=/orcd/scratch/orcd/008/merose/llc_sharding_spikes_20260903
candidate=${SLURM_ARRAY_TASK_ID:?}
"$spike_root/env/bin/python" "$spike_root/scripts/schema_bakeoff.py" \
  "$spike_root/fixtures/cody_face1_t0_t1" \
  "$spike_root/results/codec-$candidate" \
  --candidate "$candidate" --arrays U Theta --time-count 2
