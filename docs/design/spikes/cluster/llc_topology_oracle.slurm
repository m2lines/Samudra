#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

#SBATCH --job-name=llc-topology-oracle
#SBATCH --partition=pi_abodner
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
spike_root=/orcd/scratch/orcd/008/merose/llc_sharding_spikes_20260903
"$spike_root/xgcm-env/bin/python" "$spike_root/scripts/llc_topology_oracle.py" \
  /orcd/data/abodner/003/LLC4320/LLC4320 \
  "$spike_root/results/all-face-pilot.icechunk" \
  "$spike_root/results/all-face-pilot-result.json" \
  > "$spike_root/results/topology-oracle-result.json"
