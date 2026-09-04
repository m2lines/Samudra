#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

#SBATCH --job-name=llc-read-concurrency
#SBATCH --partition=pi_abodner
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G

set -euo pipefail
export BLOSC_NTHREADS=1
spike_root=/orcd/scratch/orcd/008/merose/llc_sharding_spikes_20260903
"$spike_root/env/bin/python" "$spike_root/scripts/concurrency_sweep.py" \
  "$spike_root/results/xarray-pilot.icechunk" \
  "$spike_root/results/xarray-pilot-result.json" \
  "$spike_root/results/concurrency-sweep.json"
