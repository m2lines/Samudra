#!/bin/bash -l
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=B:USR1@300
set -euo pipefail
root=$1
code=$2
stage=$3
export SAMUDRA_CODE_COMMIT=$4
export SAMUDRA_RUNTIME_SHA256
SAMUDRA_RUNTIME_SHA256=$(sha256sum "$root/runtime-contract.json" | cut -d ' ' -f1)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 WANDB_MODE=offline
export APPTAINER_CACHEDIR=/scratch/jr7309/apptainer-cache
export APPTAINER_TMPDIR=${SLURM_TMPDIR:-/tmp}
if [[ "$stage" == qualify ]]; then
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
fi
arms=(U-multitask-early U-multitask-early-latent U-multitask-early-fine U-multitask-early-fine-latent U-global U-global-seed1730 U-global-repro4k)
arm=${arms[${SLURM_ARRAY_TASK_ID:-0}]}
image=/scratch/jr7309/.apptainer-images/ghcr.io-m2lines-ocean-emulator-physicsnemo-26.05-9cd36b1bdcf4921fb027494afa4157e217818ccd.sif
binds="/scratch:/scratch:ro,$root:$root:rw,$code:/pilot-code:ro"
if [[ -n ${SLURM_TMPDIR:-} ]]; then
  binds+=",$SLURM_TMPDIR:$SLURM_TMPDIR:rw"
fi
flags=()
if [[ "$stage" != prepare ]]; then
  flags+=(--nv)
  # Torch cgroups renumber allocated devices; do not pass physical SLURM GPU IDs.
  nvidia-smi --query-gpu=timestamp,uuid,index,utilization.gpu,memory.used --format=csv -l 5 > "$root/logs/gpu-${SLURM_JOB_ID}.csv" &
  telemetry=$!
  trap 'kill "$telemetry" 2>/dev/null || true' EXIT
fi
trap 'scancel --signal=USR1 "$SLURM_JOB_ID.0" || true' USR1
srun /share/apps/apptainer/bin/apptainer exec "${flags[@]}" --bind "$binds" "$image" \
 env PYTHONPATH="/pilot-code/src:/pilot-code:/scratch/jr7309/runs/2026-10-08-early-fine-rtx/runtime" \
 /workspace/.venv/bin/python /pilot-code/scripts/early_fine_supervisor.py \
 --root "$root" --stage "$stage" --arm "$arm" &
child=$!
set +e
wait "$child"
status=$?
while kill -0 "$child" 2>/dev/null; do
  wait "$child"
  status=$?
done
exit "$status"
