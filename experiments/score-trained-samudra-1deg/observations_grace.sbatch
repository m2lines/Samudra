#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --partition=grace
#SBATCH --qos=standard
#SBATCH --account=ny_lz1955_multiscale
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=06:00:00

set -euo pipefail
export CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-/projects/ny/lz1955/multiscale/jrusak/runs/score-samudra-1deg-20261011}"
: "${SOURCE_ROOT:?Set SOURCE_ROOT to an immutable source snapshot}"
: "${FOLLOWUP_CREDENTIAL:?Set FOLLOWUP_CREDENTIAL to the private callback file}"
: "${EXPORT_ROOT:?Set EXPORT_ROOT to the verified complete trajectory export}"
on_exit() {
    rc=$?
    trap - EXIT
    python3 "$CAMPAIGN_ROOT/private/send.py" --credential-file "$FOLLOWUP_CREDENTIAL" || true
    exit "$rc"
}
trap on_exit EXIT
export NVIDIA_VISIBLE_DEVICES=void
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export WANDB_MODE=online
export SAMUDRA_CODE_COMMIT="$(cat "$SOURCE_ROOT/COMMIT")"
export SAMUDRA_CONTAINER_IMAGE_REF=ghcr.io/m2lines/ocean-emulator-physicsnemo@sha256:7d78e8fb9592abdf046d88e55004f15cfe34f0c8ee38b17d83e98cc05b6f82c4
export EVAL_OUTPUT="$CAMPAIGN_ROOT/outputs/observations-$SLURM_JOB_ID"
mkdir -p "$CAMPAIGN_ROOT/allocations"
scontrol show job "$SLURM_JOB_ID" > "$CAMPAIGN_ROOT/allocations/$SLURM_JOB_ID.txt"
python3 "$CAMPAIGN_ROOT/private/send.py" --credential-file "$FOLLOWUP_CREDENTIAL" --check || true
mounts="$CAMPAIGN_ROOT:$CAMPAIGN_ROOT,$SOURCE_ROOT/src:/workspace/src,/projects/ny/lz1955/multiscale/jrusak/data:/projects/ny/lz1955/multiscale/jrusak/data"
srun --input=none --kill-on-bad-exit=1 --container-env=NVIDIA_VISIBLE_DEVICES --container-image="$SAMUDRA_CONTAINER_IMAGE_REF" --container-mount-home --container-mounts="$mounts" bash -c '
set -euo pipefail
export PYTHONPATH=/workspace/src
cd "$SOURCE_ROOT/experiments/score-trained-samudra-1deg"
exec /workspace/.venv/bin/python observations.py "$CAMPAIGN_ROOT" "$EXPORT_ROOT" "$EVAL_OUTPUT"
'
