#!/bin/bash -l
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=diffusion-obs-cooldown
#SBATCH --account=ny_lz1955_multiscale
#SBATCH --partition=beta
#SBATCH --qos=standard
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=144
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --segment=2
#SBATCH --time=16:00:00
set -euo pipefail
: "${CAMPAIGN_ROOT:?}"
: "${CODE_ID:?}"
: "${COOLDOWN_PARENT:?}"
: "${COOLDOWN_OUTPUT:?}"
export SAMUDRA_CODE_COMMIT="$CODE_ID"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 TORCHINDUCTOR_COMPILE_THREADS=2
export NCCL_MNNVL_ENABLE=1 NCCL_DEBUG=INFO NCCL_SOCKET_IFNAME=bond0
export MASTER_ADDR=$(hostname -i)
export MASTER_PORT=29618
export NCCL_DEBUG_FILE="$CAMPAIGN_ROOT/logs/cooldown-nccl-$SLURM_JOB_ID-%h-%p.log"
code="$CAMPAIGN_ROOT/code/$CODE_ID"
image=ghcr.io/m2lines/ocean-emulator-physicsnemo@sha256:7d78e8fb9592abdf046d88e55004f15cfe34f0c8ee38b17d83e98cc05b6f82c4 # pragma: allowlist secret
container=(--container-image="$image" --container-mount-home --container-mounts="$code:/pilot-code,$CAMPAIGN_ROOT/native:/pilot-native,/projects/ny/lz1955/multiscale:/projects/ny/lz1955/multiscale")
srun --kill-on-bad-exit=1 "${container[@]}" env CC=/usr/bin/gcc CXX=/usr/bin/g++ PYTHONPATH=/pilot-code/src:/pilot-native \
  bash /pilot-code/scripts/diffusion_obs_cooldown_beta_rank.sh
