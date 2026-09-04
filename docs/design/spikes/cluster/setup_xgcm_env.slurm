#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

#SBATCH --job-name=llc-xgcm-env
#SBATCH --partition=mit_quicktest
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail
spike_root=/orcd/scratch/orcd/008/merose/llc_sharding_spikes_20260903
if [ -e "$spike_root/xgcm-env" ]; then
  echo "refusing to overwrite $spike_root/xgcm-env" >&2
  exit 1
fi
cp -a "$spike_root/env" "$spike_root/xgcm-env"
"$spike_root/xgcm-env/bin/python" -m pip install 'xgcm==0.9.0'
"$spike_root/xgcm-env/bin/python" -c \
  'import xgcm; print(xgcm.__version__)'
