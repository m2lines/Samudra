#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

#SBATCH --job-name=llc-surface-time-fixture
#SBATCH --partition=pi_abodner
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail
spike_root=/orcd/scratch/orcd/008/merose/llc_sharding_spikes_20260903
"$spike_root/env/bin/python" \
  "$spike_root/scripts/extract_timeseries_fixture.py" \
  /orcd/data/abodner/003/LLC4320/LLC4320 \
  "$spike_root/fixtures/face1_eta_t0_t23_full" \
  --time-count 24 --j-start 0 --i-start 0 --width 4320 --arrays Eta
