#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

#SBATCH --job-name=llc-range-server
#SBATCH --partition=mit_quicktest
#SBATCH --time=00:12:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G

set -euo pipefail
spike_root=/orcd/scratch/orcd/008/merose/llc_remote_streaming_spike_20260908
python=/orcd/scratch/orcd/008/merose/llc_sharding_spikes_20260903/env/bin/python
object=/orcd/scratch/orcd/008/merose/llc_sharding_spikes_20260903/results/all-face-pilot.icechunk/chunks/1GFY6VA57P9VZ45PDM1G
mkdir -p "$spike_root/server-$SLURM_JOB_ID"
"$python" "$spike_root/range_server.py" \
  "$object" "$spike_root/server-$SLURM_JOB_ID" \
  --port 18080 --duration 660
