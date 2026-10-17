#!/bin/bash -l
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=monthly-latent-qual
#SBATCH --account=ny_lz1955_multiscale
#SBATCH --partition=beta_test
#SBATCH --qos=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=192G
#SBATCH --time=01:00:00
set -euo pipefail
: "${PILOT_ROOT:?}"
: "${PRODUCER:?}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
code="$PILOT_ROOT/code/$PRODUCER"
args=()
if [[ "${FULL_QUALIFICATION:-0}" == 1 ]]; then
  args+=(--om4 /projects/ny/lz1955/multiscale/jrusak/data/diffusion-interior-beta/data/om4_onedeg_v3
    --cm4 /projects/ny/lz1955/multiscale/jrusak/data/cm4_onedeg/CM4.zarr
    --cm4-verification /projects/ny/lz1955/multiscale/jrusak/transfers/2026-10-16-cm4-pilot/COMPLETE.json)
fi
if [[ -n "${PACKED_OM4:-}" ]]; then args+=(--packed-om4 "$PACKED_OM4"); fi
image=ghcr.io/m2lines/ocean-emulator-physicsnemo@sha256:7d78e8fb9592abdf046d88e55004f15cfe34f0c8ee38b17d83e98cc05b6f82c4 # pragma: allowlist secret
scontrol show job "$SLURM_JOB_ID" > "$PILOT_ROOT/logs/allocation-$SLURM_JOB_ID.txt"
srun --cpus-per-task="$SLURM_CPUS_PER_TASK" --kill-on-bad-exit=1 --container-image="$image" --container-mount-home \
  --container-mounts="$code:/pilot-code:ro,/projects/ny/lz1955/multiscale:/projects/ny/lz1955/multiscale" \
  env PYTHONPATH=/pilot-code/src /workspace/.venv/bin/python /pilot-code/scripts/qualify_monthly_latent.py \
  --observations /projects/ny/lz1955/multiscale/jrusak/data/obs_full_range/d-observation-pilot/samples \
  --output "$PILOT_ROOT/initial-qualification-$SLURM_JOB_ID" --producer "$PRODUCER" "${args[@]}"
