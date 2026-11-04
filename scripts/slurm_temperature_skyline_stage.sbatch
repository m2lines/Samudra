#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=skyline-stage
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail
: "${SKYLINE_ROOT:?}" "${SKYLINE_COMMIT:?}" "${SKYLINE_ARCHIVE_SHA256:?}"
: "${SKYLINE_CONFIG:?}" "${SKYLINE_OUTPUT:?}" "${SIF_PATH:?}"
if [[ -f "${SKYLINE_OUTPUT}/COMPLETE.json" ]]; then exit 0; fi
requeue_job() { scontrol requeue "$SLURM_JOB_ID"; }
trap requeue_job USR1
export APPTAINER_CACHEDIR=/scratch/jr7309/apptainer-cache
export SINGULARITY_CACHEDIR=/scratch/jr7309/singularity-cache
export APPTAINER_TMPDIR="${SLURM_TMPDIR:-/tmp}"
export PYTHONUNBUFFERED=1
export WANDB_DIR="${SKYLINE_OUTPUT}/wandb"
export WANDB_CACHE_DIR="${SKYLINE_ROOT}/wandb-cache"
export WANDB_CONFIG_DIR="${SKYLINE_ROOT}/wandb-config"
mkdir -p "$SKYLINE_OUTPUT" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"
# The scheduler, rather than a rollback-prone checkpoint timer, charges prior
# allocations. Replayed updates and failed allocations remain in this budget.
read -r SKYLINE_PRIOR_SECONDS SKYLINE_ALLOCATION_START < <(python3 - <<'PY'
import datetime
import os
import subprocess
job = os.environ['SLURM_JOB_ID']
current = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)
fields = dict(item.split('=', 1) for item in current.split() if '=' in item)
start = datetime.datetime.fromisoformat(fields['StartTime']).timestamp()
raw = subprocess.check_output(['sacct', '-j', job, '-X', '--duplicates', '-n', '-P',
                               '-o', 'JobIDRaw,Start,End,ElapsedRaw'], text=True)
prior = 0
for line in raw.splitlines():
    jid, begin, end, seconds, *_ = line.split('|')
    if jid != job or begin in {'Unknown', 'None'}:
        continue
    epoch = datetime.datetime.fromisoformat(begin).timestamp()
    if epoch < start:
        if end in {'Unknown', 'None'}:
            raise RuntimeError('Prior allocation accounting is incomplete')
        prior += int(seconds)
print(prior, start)
PY
)
export SKYLINE_PRIOR_SECONDS SKYLINE_ALLOCATION_START
printf 'Prior allocated seconds: %s; current allocation began: %s\n' "$SKYLINE_PRIOR_SECONDS" "$SKYLINE_ALLOCATION_START"
ARCHIVE="${SKYLINE_ROOT}/source-${SKYLINE_COMMIT}.tar"
actual=$(sha256sum "$ARCHIVE" | cut -d ' ' -f 1)
test "$actual" = "$SKYLINE_ARCHIVE_SHA256"
CODE_DIR=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/skyline-stage.XXXXXX")
trap 'rm -rf "$CODE_DIR"' EXIT
tar -xf "$ARCHIVE" -C "$CODE_DIR"
nvidia-smi
printf 'W&B mode: %s\nSource commit: %s\n' "${WANDB_MODE:-online}" "$SKYLINE_COMMIT"
apptainer exec --nv --bind "${CODE_DIR}:/opt/skyline-source:ro,/scratch:/scratch" \
    "$SIF_PATH" /usr/bin/env PYTHONPATH=/opt/skyline-source/src \
    TORCH_CUDNN_V8_API_DISABLED=1 OMP_NUM_THREADS=4 \
    /workspace/.venv/bin/torchrun --standalone --nnodes=1 --nproc-per-node="${SKYLINE_WORLD_SIZE:-8}" \
    -m samudra.skyline.stage --config "$SKYLINE_CONFIG" --output "$SKYLINE_OUTPUT" &
child=$!
while true; do
    status=0
    wait "$child" || status=$?
    if ! kill -0 "$child" 2>/dev/null; then exit "$status"; fi
done
