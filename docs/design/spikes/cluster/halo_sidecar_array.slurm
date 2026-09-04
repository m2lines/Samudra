#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

#SBATCH --job-name=llc-halo-control
#SBATCH --partition=pi_abodner
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --array=0-1%2

set -euo pipefail
export BLOSC_NTHREADS=1
spike_root=/orcd/scratch/orcd/008/merose/llc_sharding_spikes_20260903
halos=(16 32)
halo=${halos[${SLURM_ARRAY_TASK_ID:?}]}
candidate=$("$spike_root/env/bin/python" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["winner"]["candidate_index"])' \
  "$spike_root/results/codec-winner.json")
"$spike_root/env/bin/python" "$spike_root/scripts/halo_sidecar_bakeoff.py" \
  "$spike_root/fixtures/cody_face1_t0_t1" \
  "$spike_root/results/halo-$halo" \
  --candidate "$candidate" --halo "$halo"
