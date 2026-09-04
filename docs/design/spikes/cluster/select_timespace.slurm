#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

#SBATCH --job-name=llc-pick-timespace
#SBATCH --partition=mit_quicktest
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail
spike_root=/orcd/scratch/orcd/008/merose/llc_sharding_spikes_20260903
"$spike_root/env/bin/python" "$spike_root/scripts/select_timespace.py" \
  "$spike_root/results/timespace-*/result.json" \
  "$spike_root/results/timespace-winner.json"
