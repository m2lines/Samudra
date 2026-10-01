#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=obs-D-coarsen
#SBATCH --account=ny_lz1955_multiscale
#SBATCH --partition=grace
#SBATCH --qos=standard
#SBATCH --exclude=alphagh01
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-7
set -euo pipefail
root=/mnt/home/jrusak/data/obs_full_range
export PATH="$root/envs/grace-arm64/bin:$HOME/bin:$PATH"
export PYTHONPATH="$root/d-observation-pilot/code/deps"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLOSC_NTHREADS=1
products=(sst adt era5 iap)
for ((id=SLURM_ARRAY_TASK_ID; id<=122; id+=8)); do
limit=0
if ((id < 120)); then
  product=${products[$((id / 30))]}
  year=$((1993 + id % 30))
else
  product=${products[$((id - 120))]}
  year=2023
  limit=5
fi
cd "$root/d-observation-pilot/code/frozen-preparation"
sha256sum --check SHA256SUMS
python observation_prepare.py --source "$root/prepared" --grid grid.npz --output "$root/d-observation-pilot/daily" --product "$product" --year "$year" --limit "$limit"

done
