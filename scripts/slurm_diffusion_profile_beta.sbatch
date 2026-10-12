#!/bin/bash -l
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=diffusion-full-profile
#SBATCH --account=ny_lz1955_multiscale
#SBATCH --partition=beta_test
#SBATCH --qos=test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=144
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --segment=2
#SBATCH --time=01:00:00
set -euo pipefail
: "${PROFILE_ROOT:?}"
export SAMUDRA_CODE_COMMIT=${PROFILE_CODE_ID:?}
export DATA_ROOT=/projects/ny/lz1955/multiscale/jrusak/runs/diffusion-observation-fast-v1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 TORCHINDUCTOR_COMPILE_THREADS=2 WANDB_MODE=disabled
export NCCL_MNNVL_ENABLE=1 NCCL_DEBUG=INFO
# The automatic ibp3s0 bootstrap selection was unroutable between test hosts.
export NCCL_SOCKET_IFNAME=bond0
export MASTER_ADDR=$(hostname -i)
export MASTER_PORT=29612
export PROFILE_OUTPUT="$PROFILE_ROOT/results/$SLURM_JOB_ID"
mkdir -p "$PROFILE_OUTPUT"
scontrol show job "$SLURM_JOB_ID" > "$PROFILE_OUTPUT/allocation.txt"
image=ghcr.io/m2lines/ocean-emulator-physicsnemo@sha256:7d78e8fb9592abdf046d88e55004f15cfe34f0c8ee38b17d83e98cc05b6f82c4 # pragma: allowlist secret
container=(--container-image="$image" --container-mount-home --container-mounts="$PROFILE_ROOT/code:/profile-code,$DATA_ROOT/native:/pilot-native,/projects/ny/lz1955/multiscale:/projects/ny/lz1955/multiscale")
srun --kill-on-bad-exit=1 "${container[@]}" env CC=/usr/bin/gcc CXX=/usr/bin/g++ PYTHONPATH=/profile-code/src:/pilot-native bash /profile-code/scripts/profile_diffusion_beta_rank.sh single
srun --kill-on-bad-exit=1 "${container[@]}" env CC=/usr/bin/gcc CXX=/usr/bin/g++ PYTHONPATH=/profile-code/src:/pilot-native bash /profile-code/scripts/profile_diffusion_beta_rank.sh distributed
