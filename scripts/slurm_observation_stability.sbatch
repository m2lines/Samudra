#!/bin/bash -l
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
set -euo pipefail
root=$1
code=$2
stage=$3
export SAMUDRA_CODE_COMMIT=$4
export SAMUDRA_RUNTIME_SHA256
SAMUDRA_RUNTIME_SHA256=$(sha256sum "$root/runtime-contract.json" | cut -d ' ' -f1)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 WANDB_MODE=online
export APPTAINER_CACHEDIR=/scratch/jr7309/apptainer-cache
export APPTAINER_TMPDIR=${SLURM_TMPDIR:-/tmp}
export REQUEUE_ON_USR1=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
arms=(U-global U-multitask-early)
index=${SLURM_ARRAY_TASK_ID:-0}
arm=${arms[$((index % 2))]}
trial=$((index / 2))
image=/scratch/jr7309/.apptainer-images/ghcr.io-m2lines-ocean-emulator-physicsnemo-26.05-9cd36b1bdcf4921fb027494afa4157e217818ccd.sif
binds="/scratch:/scratch:ro,$root:$root:rw,$code:/pilot-code:ro"
if [[ -n ${SLURM_TMPDIR:-} ]]; then
  binds+=",$SLURM_TMPDIR:$SLURM_TMPDIR:rw"
fi
# Diagnostic subprocesses cannot mutate the shared treatment forks.
if [[ "$stage" == trace || "$stage" == repeat ]]; then
  binds+=",$root/$arm/fork:$root/$arm/fork:ro"
fi
nvidia-smi --query-gpu=timestamp,uuid,index,utilization.gpu,memory.used --format=csv -l 5 > "$root/logs/gpu-${SLURM_JOB_ID}-${SLURM_RESTART_COUNT:-0}.csv" &
telemetry=$!
trap 'kill "$telemetry" 2>/dev/null || true' EXIT
stopping=0
trap 'stopping=1; scancel --signal=USR1 "$SLURM_JOB_ID.0" || true' USR1
srun /share/apps/apptainer/bin/apptainer exec --nv --bind "$binds" "$image" \
 env PYTHONPATH="/pilot-code/src:/pilot-code:/scratch/jr7309/runs/2026-10-08-early-fine-rtx/runtime" \
 /workspace/.venv/bin/python -m scripts.run_observation_stability \
 --root "$root" --stage "$stage" --arm "$arm" --trial "$trial" &
child=$!
set +e
wait "$child"
status=$?
while kill -0 "$child" 2>/dev/null; do
  wait "$child"
  status=$?
done
set -e
if [[ "$status" == 75 || "$stopping" == 1 ]]; then
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$status"
