#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

#SBATCH --job-name=llc-range-client
#SBATCH --account=nyu
#SBATCH --partition=cpu
#SBATCH --qos=test
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G

set -euo pipefail
spike_root=${LLC_SPIKE_ROOT:?Set LLC_SPIKE_ROOT to the Empire AI spike directory}
python3 "$spike_root/range_client.py" \
  "$spike_root/endpoint.json" "$spike_root/token" \
  > "$spike_root/result-$SLURM_JOB_ID.json"
