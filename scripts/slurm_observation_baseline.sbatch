#!/bin/bash -l
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=observation-baseline
#SBATCH --account=ny_lz1955_multiscale
#SBATCH --partition=beta
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=144
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@180

# Submit held with --output="$RUN_ROOT/slurm-%j.log"; install the private
# callback before releasing. Source must already be frozen in RUN_ROOT/code.
set -euo pipefail
: "${RUN_ROOT:?Set RUN_ROOT to the campaign output directory}"
: "${CALLBACK_DIR:?Set CALLBACK_DIR to its private callback directory}"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 WANDB_MODE=online

finish() {
    rc=$?
    trap - EXIT
    python3 "$CALLBACK_DIR/send.py" --credential-file "$CALLBACK_DIR/callback.json" || true
    exit "$rc"
}
trap finish EXIT
python3 "$CALLBACK_DIR/send.py" --credential-file "$CALLBACK_DIR/callback.json" --check
scontrol show job "$SLURM_JOB_ID" > "$RUN_ROOT/allocation-$SLURM_JOB_ID.txt"
# srun forwards USR1 to the supervisor, which asks trainers to save and stop.
# Bash's wait is interrupted by a trapped signal, so wait again until srun exits.
launcher_pid=
request_stop() {
    if [[ -n "$launcher_pid" ]]; then
        kill -USR1 "$launcher_pid" || true
    fi
}
trap request_stop USR1 TERM INT
srun --kill-on-bad-exit=1 \
    --container-image=ghcr.io/m2lines/ocean-emulator-physicsnemo@sha256:7d78e8fb9592abdf046d88e55004f15cfe34f0c8ee38b17d83e98cc05b6f82c4 \
    --container-mount-home \
    --container-mounts="$RUN_ROOT/code:/baseline-code:ro,/projects/ny/lz1955/multiscale:/projects/ny/lz1955/multiscale" \
    env PYTHONPATH=/baseline-code/src \
    /workspace/.venv/bin/python /baseline-code/scripts/run_observation_baseline_campaign.py \
    --output "$RUN_ROOT" "$@" &
launcher_pid=$!
rc=0
while true; do
    if wait "$launcher_pid"; then rc=0; else rc=$?; fi
    if ! kill -0 "$launcher_pid" 2>/dev/null; then break; fi
done
exit "$rc"
