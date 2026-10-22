#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=cm4-century-viz
#SBATCH --partition=grace
#SBATCH --account=ny_lz1955_multiscale
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=384G
#SBATCH --time=11:55:00
set -euo pipefail
export ROOT=/projects/ny/lz1955/multiscale/jrusak/runs/cm4-onedeg-samudra-20261017/viz-100yr-20261022-attempt2
# The pinned NVIDIA image also serves CPU analysis: do not request driver
# injection on Grace nodes, which have no NVIDIA GPU driver.
export NVIDIA_VISIBLE_DEVICES=void
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="$ROOT/cache/matplotlib" SAMUDRA_CODE_COMMIT="$(cat "$ROOT/commit.txt")"
export SAMUDRA_CONTAINER_IMAGE_REF=ghcr.io/m2lines/ocean-emulator-physicsnemo@sha256:7d78e8fb9592abdf046d88e55004f15cfe34f0c8ee38b17d83e98cc05b6f82c4
on_exit() {
    rc=$?
    trap - EXIT
    python3 "$ROOT/private/send.py" --credential-file "$ROOT/private/completion.json" || true
    exit "$rc"
}
trap on_exit EXIT
mkdir -p "$MPLCONFIGDIR"
scontrol show job "$SLURM_JOB_ID" > "$ROOT/allocation.txt"
python3 "$ROOT/private/send.py" --credential-file "$ROOT/private/completion.json" --check
srun --kill-on-bad-exit=1 --container-image="$SAMUDRA_CONTAINER_IMAGE_REF" --container-mount-home --container-env=NVIDIA_VISIBLE_DEVICES \
    --container-mounts="$ROOT/source/src:/workspace/src,/projects/ny/lz1955/multiscale:/projects/ny/lz1955/multiscale,$ROOT/cache:/workspace/.data_cache" \
    bash -c '
set -euo pipefail
export PYTHONPATH=/workspace/src
cd "$ROOT"
command -v ffmpeg
command -v ffprobe
exec /workspace/.venv/bin/python "$ROOT/source/experiments/cm4-onedeg-samudra/run_viz.py" "$ROOT/viz.yaml"
'
