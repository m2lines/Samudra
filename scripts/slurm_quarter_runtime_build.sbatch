#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
# Submit through Alpha only, after explicit review; no GPUs or model execution.
#SBATCH --job-name=quarter-runtime-build
#SBATCH --account=ny_lz1955_multiscale
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00

# ROOT/source is an immutable reviewed git archive; source.sha256 is produced
# locally from `find source -type f -print0 | sort -z | xargs -0 sha256sum`.
# ROOT itself is dedicated quarter preparation project storage. Each attempt
# writes ROOT/runtime-build/job-ID and never changes prior runtime artifacts.
set -euo pipefail
root=${1:?Pass reviewed quarter project-storage root}
data_root=/projects/ny/lz1955/multiscale/jrusak/data/om4/v2026-09/om4_quarterdeg_compact_zstd
image=ghcr.io/m2lines/ocean-emulator-physicsnemo@sha256:7d78e8fb9592abdf046d88e55004f15cfe34f0c8ee38b17d83e98cc05b6f82c4
test -f "$root/source.sha256"
test -f "$root/source/scripts/build_quarter_runtime.sh"
test -f "$data_root/OM4.zarr/.zmetadata"
(cd "$root" && sha256sum --check source.sha256)
mkdir -p "$root/runtime-build"
output="$root/runtime-build/job-$SLURM_JOB_ID"
test ! -e "$output"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 WANDB_MODE=disabled
# Grace CPU nodes do not load NVIDIA drivers. The Enroot NVIDIA hook explicitly
# skips NVML initialization for "void"; pass it through to the container too.
export NVIDIA_VISIBLE_DEVICES=void
scontrol show job "$SLURM_JOB_ID" > "$root/runtime-build/allocation-$SLURM_JOB_ID.txt"
printf '%s\n' "$image" > "$root/runtime-build/image-$SLURM_JOB_ID.txt"
srun --kill-on-bad-exit=1 --container-image="$image" \
  --container-env=NVIDIA_VISIBLE_DEVICES \
  --container-mounts="$root/source:/source:ro,$root/runtime-build:/build:rw,$data_root:/data:ro" \
  --container-workdir=/source \
  bash /source/scripts/build_quarter_runtime.sh /source "/build/job-$SLURM_JOB_ID" /data/OM4.zarr
