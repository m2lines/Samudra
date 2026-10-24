#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=s3-transfer-verify
#SBATCH --account=torch_pr_347_lzanna
#SBATCH --partition=cs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --chdir=/scratch/jr7309/benchmarks/quarterdeg-local-s3-20261021
#SBATCH --output=/scratch/jr7309/benchmarks/quarterdeg-local-s3-20261021/ready-%j.out
#SBATCH --error=/scratch/jr7309/benchmarks/quarterdeg-local-s3-20261021/ready-%j.err
set -euo pipefail
base=$PWD
test -f "$base/BUILD_COMPLETE"
while [[ ! -f "$base/TRANSFER_COMPLETE.json" ]]; do sleep 60; done
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$base/extension:$base/repo/src:$base/repo"
apptainer exec --bind "$HOME/.config/rclone/rclone.conf:/opt/osn-rclone.conf:ro" \
  /scratch/jr7309/.apptainer-images/ghcr.io-m2lines-ocean-emulator-physicsnemo-26.05-9cd36b1bdcf4921fb027494afa4157e217818ccd.sif \
  /workspace/.venv/bin/python "$base/benchmark.py" --preflight
