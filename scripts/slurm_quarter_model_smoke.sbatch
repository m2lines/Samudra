#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=quarter-model-smoke
#SBATCH --account=ny_lz1955_multiscale
#SBATCH --partition=beta_test
#SBATCH --qos=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=01:00:00

# Prepared only: caller must review/stage source, manifest and this script.
# ROOT layout: source/src from git archive 56aac937; U-global-manifest.json;
# quarter_model_smoke.py. Outputs are diagnostic JSON only, never checkpoints.
set -euo pipefail
root=${1:?Pass the reviewed project-storage smoke directory}
test -s "$root/U-global-manifest.json"
test -s "$root/source/src/samudra/experiments/observation_model.py"
test -s "$root/quarter_model_smoke.py"
image=ghcr.io/m2lines/ocean-emulator-physicsnemo@sha256:7d78e8fb9592abdf046d88e55004f15cfe34f0c8ee38b17d83e98cc05b6f82c4
export QUARTER_SMOKE_IMAGE="$image"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 WANDB_MODE=disabled
output="$root/job-$SLURM_JOB_ID"
mkdir "$output"
scontrol show job "$SLURM_JOB_ID" > "$output/allocation.txt"
status=0
for case in om4-6 observation-6 observation-7; do
  task=${case%-*}
  leads=${case##*-}
  # Each case gets a fresh process/model/allocator. Attempt all three cases and
  # return failure if any fails; failed JSON/tracebacks remain for diagnosis.
  if ! srun --kill-on-bad-exit=1 --container-image="$image" \
    --container-mounts="$root:/diag:ro,$output:/diag/job-$SLURM_JOB_ID:rw" --container-workdir=/diag \
    env PYTHONPATH=/diag/source/src \
    /workspace/.venv/bin/python /diag/quarter_model_smoke.py \
    --source /diag/source --manifest /diag/U-global-manifest.json \
    --output "/diag/job-$SLURM_JOB_ID/$case" --task "$task" --leads "$leads" \
    > "$output/$case.log" 2>&1; then
    status=1
  fi
done
exit "$status"
