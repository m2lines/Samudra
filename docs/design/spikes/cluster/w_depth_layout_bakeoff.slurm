#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

#SBATCH --job-name=llc-w-depth
#SBATCH --partition=pi_abodner
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
export BLOSC_NTHREADS=1
spike_root=/orcd/scratch/orcd/008/merose/llc_sharding_spikes_20260903
"$spike_root/env/bin/python" "$spike_root/scripts/w_depth_layout_bakeoff.py" \
  /orcd/data/abodner/003/LLC4320/LLC4320 \
  "$spike_root/results/w-depth-layout" \
  > "$spike_root/results/w-depth-layout.json"
