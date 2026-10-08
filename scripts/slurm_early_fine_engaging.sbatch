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
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
image=${SAMUDRA_IMAGE:-/orcd/scratch/orcd/014/jrusak/apptainer_cache/images/diffusion-26.05-9cd36b1-amd64.sif}
data_bind=${SAMUDRA_DATA_BIND:-/orcd:/orcd}
if [[ -n ${SLURM_TMPDIR:-} ]]; then
  data_bind+=",$SLURM_TMPDIR:$SLURM_TMPDIR"
fi
if ! command -v apptainer >/dev/null; then
  module load apptainer/1.5.2
fi
arms=(U-multitask-early U-multitask-early-latent U-multitask-early-fine U-multitask-early-fine-latent)
arm=${arms[${SLURM_ARRAY_TASK_ID:-0}]}
export SAMUDRA_RUNTIME_SHA256
SAMUDRA_RUNTIME_SHA256=$(sha256sum "$root/runtime-contract.json" | cut -d ' ' -f1)
# Forward the warning to the Python supervisor, which forwards to its trainer.
trap 'scancel --signal=USR1 "$SLURM_JOB_ID.0" || true' USR1
srun apptainer exec ${GPU_ENABLED:+--nv} --bind "$code:/pilot-code,$data_bind" "$image" \
 env PYTHONPATH="/pilot-code/src:/pilot-code:$root/runtime" \
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
set -e
if [[ "$status" == 75 ]]; then
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$status"
