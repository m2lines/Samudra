#!/bin/bash -l
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=stochastic-replay
#SBATCH --account=ny_lz1955_multiscale
#SBATCH --partition=beta
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=144
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=08:00:00
set -euo pipefail
: "${CAMPAIGN_ROOT:?}"
: "${CODE_ID:?}"
: "${MODE:?qualify or train}"
: "${REPLAY_CACHE:?existing verified monthly replay cache}"
export SAMUDRA_CODE_COMMIT="$CODE_ID"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=2
code="$CAMPAIGN_ROOT/code/$CODE_ID"
image=ghcr.io/m2lines/ocean-emulator-physicsnemo@sha256:7d78e8fb9592abdf046d88e55004f15cfe34f0c8ee38b17d83e98cc05b6f82c4 # pragma: allowlist secret
scontrol show job "$SLURM_JOB_ID" > "$CAMPAIGN_ROOT/logs/allocation-$SLURM_JOB_ID.txt"
srun --kill-on-bad-exit=1 --container-image="$image" --container-mount-home \
  --container-mounts="$code:/pilot-code,$CAMPAIGN_ROOT/native:/pilot-native,/projects/ny/lz1955/multiscale:/projects/ny/lz1955/multiscale" \
  env CC=/usr/bin/gcc CXX=/usr/bin/g++ PYTHONPATH=/pilot-code/src:/pilot-native \
  bash /pilot-code/scripts/stochastic_replay_beta_workers.sh
