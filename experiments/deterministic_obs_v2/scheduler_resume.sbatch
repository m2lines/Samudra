#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=obs-scheduler-resume
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00

set -euo pipefail
: "${OUTPUT_BASE:?}" "${CONFIG:?}" "${RESUME_AUDIT_SCRIPT:?}" "${TRAIN_HARNESS:?}"
: "${SIF_PATH:?}" "${CODE_LAYER:?}" "${GPUS_PER_NODE:?}"

handle_requeue() {
  echo "Qualification: USR1 received; requeueing ${SLURM_JOB_ID}."
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
}
trap handle_requeue USR1

mkdir -p "$OUTPUT_BASE"
marker="$OUTPUT_BASE/requeue-request.json"
if (( ${SLURM_RESTART_COUNT:-0} > 0 )) && [[ ! -f "$marker" ]]; then
  echo "Unplanned restart: cannot establish an uninterrupted control." >&2
  exit 2
fi

export SAMUDRA_MODULE=samudra.observations.train
export SAMUDRA_MANAGE_RUN_DIR=0
export ARGS=""
if [[ ! -f "$OUTPUT_BASE/control/saved_nets/COMPLETE.json" ]]; then
  export NAME=control
  bash "$TRAIN_HARNESS" &
  wait "$!"
fi

watcher=""
if [[ ! -f "$marker" ]]; then
  python3 "$RESUME_AUDIT_SCRIPT" request "$OUTPUT_BASE" &
  watcher=$!
fi
export NAME=resumed
bash "$TRAIN_HARNESS" &
wait "$!"
if [[ -n "$watcher" ]]; then
  wait "$watcher"
fi

# The audit is CPU-only; training used the shared launcher without monkeypatches.
singularity exec --overlay "${CODE_LAYER}:ro" \
  --bind /scratch/jr7309:/scratch/jr7309 --pwd /opt/samudra-code "$SIF_PATH" \
  /usr/bin/env PYTHONPATH=/opt/samudra-code/src PYTHONUNBUFFERED=1 \
  /workspace/.venv/bin/python "$RESUME_AUDIT_SCRIPT" audit "$OUTPUT_BASE"
