#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

#SBATCH --job-name=llc-fixture
#SBATCH --partition=mit_normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail

spike_root=/orcd/scratch/orcd/008/merose/llc_sharding_spikes_20260903
"$spike_root/env/bin/python" "$spike_root/scripts/extract_real_fixture.py" \
  /orcd/data/abodner/003/LLC4320/LLC4320 \
  "$spike_root/fixtures/cody_face1_t0_t1" \
  --face 1 --times 0 1
