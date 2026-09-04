#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

#SBATCH --job-name=llc-xarray-pilot
#SBATCH --partition=pi_abodner
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G

set -euo pipefail
export BLOSC_NTHREADS=1
spike_root=/orcd/scratch/orcd/008/merose/llc_sharding_spikes_20260903
"$spike_root/env/bin/python" "$spike_root/scripts/xarray_icechunk_pilot.py" \
  "$spike_root/fixtures/face1_t0_t7_region720_3600" \
  "$spike_root/results/xarray-pilot.icechunk" \
  "$spike_root/results/codec-winner.json" \
  "$spike_root/results/general-winner.json" \
  "$spike_root/results/timespace-winner.json"
