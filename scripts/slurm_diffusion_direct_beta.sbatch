#!/bin/bash -l
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=diffusion-direct
#SBATCH --account=ny_lz1955_multiscale
#SBATCH --partition=beta
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=144
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=08:00:00
set -euo pipefail
: "${CODE_ID:?Immutable source archive required}"
: "${CAMPAIGN_ROOT:?Explicit campaign root required}"
export PREVIOUS_JOBS=${PREVIOUS_JOBS:-}
export CAMPAIGN_GPU_HOUR_CAP=${CAMPAIGN_GPU_HOUR_CAP:-40}
export RESERVED_OTHER_GPU_HOURS=${RESERVED_OTHER_GPU_HOURS:-0}
root=$CAMPAIGN_ROOT
code="$root/code/$CODE_ID"
mkdir -p "$root/logs"
# Run on the host, where Slurm accounting is available, before launching CUDA.
# Finished jobs count actual allocation time; live jobs reserve their entire cap.
python3 - "$root/logs/budget-$SLURM_JOB_ID.json" <<'PY'
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def gpu_count(tres):
    values = dict(item.split("=", 1) for item in tres.split(",") if "=" in item)
    if "gres/gpu" in values:
        return int(values["gres/gpu"])
    return sum(int(value) for key, value in values.items() if key.startswith("gres/gpu:"))


def limit_hours(value):
    days, _, clock = value.rpartition("-")
    parts = clock.split(":")
    if len(parts) != 3:
        raise ValueError(f"Unrecognized time limit: {value}")
    hours, minutes, seconds = map(int, parts)
    return int(days or 0) * 24 + hours + minutes / 60 + seconds / 3600


job_ids = [job for job in os.environ["PREVIOUS_JOBS"].split(",") if job]
if len(set(job_ids)) != len(job_ids) or not all(re.fullmatch(r"[0-9]+", job) for job in job_ids):
    raise ValueError("Previous jobs must be unique integer job IDs")
current_id = os.environ["SLURM_JOB_ID"]
if current_id in job_ids:
    raise ValueError("Current job must not also appear among previous jobs")
text = subprocess.check_output([
    "sacct", "-n", "-X", "-P", "-j", ",".join(job_ids),
    "--format=JobIDRaw,State,ElapsedRaw,TimelimitRaw,AllocTRES,ReqTRES",
], text=True) if job_ids else ""
records = {}
terminal = {"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}
for line in text.splitlines():
    job, state, elapsed, limit, allocated, requested = line.split("|")[:6]
    if job not in job_ids:
        continue
    state = state.split()[0].rstrip("+")
    gpus = gpu_count(allocated or requested)
    if gpus <= 0:
        raise ValueError(f"Missing GPU allocation for {job}")
    seconds = int(elapsed)
    if state in terminal:
        basis = "actual"
    else:
        seconds = max(seconds, int(limit) * 60)
        basis = "entire_requested_limit"
    records[job] = dict(state=state, gpus=gpus, gpu_hours=gpus * seconds / 3600, basis=basis)
if set(records) != set(job_ids):
    raise ValueError(f"Missing accounting records: {set(job_ids) - set(records)}")
current = subprocess.check_output(["scontrol", "show", "job", "-o", current_id], text=True)
fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)", current))
gpus = gpu_count(fields.get("AllocTRES", "") or fields.get("ReqTRES", ""))
if gpus != 4:
    raise ValueError(f"Expected four allocated GPUs, got {gpus}")
current_max = gpus * limit_hours(fields["TimeLimit"])
cap = float(os.environ["CAMPAIGN_GPU_HOUR_CAP"])
reserved = float(os.environ["RESERVED_OTHER_GPU_HOURS"])
if not all(math.isfinite(value) for value in (cap, reserved, current_max)) or cap <= 0 or min(reserved, current_max) < 0:
    raise ValueError("Invalid budget parameters")
total = sum(record["gpu_hours"] for record in records.values()) + reserved + current_max
report = dict(utc=datetime.now(timezone.utc).isoformat(), previous=records, current_job=current_id,
              current_max_gpu_hours=current_max, reserved_other_gpu_hours=reserved,
              total_max_gpu_hours=total, cap_gpu_hours=cap, admitted=total <= cap)
Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)
if total > cap:
    raise RuntimeError(f"Campaign budget gate rejected {total:.3f} > {cap:.3f} GPU-hours")
PY
export SAMUDRA_CODE_COMMIT="$CODE_ID"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 TORCHINDUCTOR_COMPILE_THREADS=2
export WANDB_MODE=online
image=ghcr.io/m2lines/ocean-emulator-physicsnemo@sha256:7d78e8fb9592abdf046d88e55004f15cfe34f0c8ee38b17d83e98cc05b6f82c4 # pragma: allowlist secret
args=(--root "$root" --updates 128 --hours 7)
if [[ "${SMOKE:-0}" == 1 ]]; then args+=(--smoke); export WANDB_MODE=disabled; fi
scontrol show job "$SLURM_JOB_ID" > "$root/logs/allocation-$SLURM_JOB_ID.txt"
srun --kill-on-bad-exit=1 --container-image="$image" --container-mount-home \
  --container-mounts="$code:/pilot-code,$root/native:/pilot-native,/projects/ny/lz1955/multiscale:/projects/ny/lz1955/multiscale" \
  env CC=/usr/bin/gcc CXX=/usr/bin/g++ PYTHONPATH=/pilot-code/src:/pilot-native \
  /workspace/.venv/bin/python /pilot-code/scripts/run_diffusion_direct.py "${args[@]}"
