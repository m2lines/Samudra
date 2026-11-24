#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=precision-competition
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

set -euo pipefail
: "${RUN_ROOT:?}" "${SOURCE_COMMIT:?}" "${SIF_PATH:?}" "${MANIFEST_SHA256:?}"
cd "$RUN_ROOT/code"
echo "$MANIFEST_SHA256  experiments/precision_competition/manifest.json" | sha256sum -c -
export APPTAINER_CACHEDIR="$RUN_ROOT/cache"
export APPTAINER_TMPDIR="${SLURM_TMPDIR:-/tmp}"
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
RUNTIME=/orcd/software/core/001/pkg/apptainer/1.5.2/bin/apptainer
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv
if [[ "${1:-train}" == qualify ]]; then
    exec "$RUNTIME" exec --nv --bind /orcd --pwd "$RUN_ROOT/code" "$SIF_PATH" \
        /usr/bin/env TORCH_CUDNN_V8_API_DISABLED=1 PYTHONUNBUFFERED=1 \
        /workspace/.venv/bin/python -m experiments.precision_competition.qualify \
        --output "$RUN_ROOT/qualification"
fi

echo "W&B mode: ${WANDB_MODE:-online}"
preempted=0
child=""
on_signal() {
    preempted=1
    if [[ -n "$child" ]]; then kill -USR1 "$child" 2>/dev/null || true; fi
}
trap on_signal USR1
"$RUNTIME" exec --nv --bind /orcd --pwd "$RUN_ROOT/code" "$SIF_PATH" \
    /usr/bin/env TORCH_CUDNN_V8_API_DISABLED=1 PYTHONUNBUFFERED=1 \
    /workspace/.venv/bin/python -m experiments.precision_competition.benchmark \
    --manifest experiments/precision_competition/manifest.json \
    --index "${SLURM_ARRAY_TASK_ID:?}" --output "$RUN_ROOT/results" \
    --wandb-mode "${WANDB_MODE:-online}" &
child=$!
set +e
wait "$child"
rc=$?
if [[ "$preempted" == 1 ]]; then
    wait "$child"
    rc=$?
    scontrol requeue "${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}"
    exit 0
fi
exit "$rc"
