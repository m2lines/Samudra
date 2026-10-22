#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

#SBATCH --job-name=cm4-rust-qualification
#SBATCH --partition=beta_test
#SBATCH --qos=test
#SBATCH --account=ny_lz1955_multiscale
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
set -euo pipefail

bash /projects/ny/lz1955/multiscale/jrusak/runs/cm4-onedeg-samudra-20261017/source/experiments/cm4-onedeg-samudra/train.sbatch qualify
