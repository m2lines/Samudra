#!/bin/bash -l
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=diffusion-global
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
set -euo pipefail
: "${CODE_COMMIT:?Immutable producer required}"
root=/scratch/jr7309/diffusion-global-v1
code="$root/code/$CODE_COMMIT"
image=/scratch/jr7309/.apptainer-images/ghcr.io-m2lines-ocean-emulator-physicsnemo-26.05-9cd36b1bdcf4921fb027494afa4157e217818ccd.sif
export SAMUDRA_CODE_COMMIT="$CODE_COMMIT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export TORCHINDUCTOR_COMPILE_THREADS=2
export TORCHINDUCTOR_CACHE_DIR="$root/inductor-cache"
export TRITON_CACHE_DIR="$root/triton-cache"
export APPTAINER_CACHEDIR=/scratch/jr7309/apptainer-cache
export SINGULARITY_CACHEDIR=/scratch/jr7309/singularity-cache
export WANDB_MODE=disabled
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
mode=${MODE:-qualify}
args=(--root "$root" --reference /scratch/jr7309/runs/2026-09-29-observation-global/fit-conditioned/selection-reference.json --compile-decoder)
if [[ "$mode" == qualify ]]; then
  args+=(--qualify --output "$root/qualification-$CODE_COMMIT")
elif [[ "$mode" == train ]]; then
  args+=(--output "$root/train" --qualification "$root/qualification-$CODE_COMMIT/QUALIFIED.json" --invocation-hours "${INVOCATION_HOURS:-11.5}" --checkpoint-every 10)
else
  : "${EVAL_CHECKPOINT:?Evaluation checkpoint required}"
  args+=(--output "$root/evaluation" --evaluate "$EVAL_CHECKPOINT" --eval-steps "${EVAL_STEPS:-32}")
fi
mkdir -p "$root/inductor-cache" "$root/triton-cache"
child=
requeue_requested=0
request_requeue() {
  requeue_requested=1
  if [[ -n "$child" ]]; then kill -TERM "$child" 2>/dev/null || true; fi
}
trap request_requeue USR1
srun apptainer exec --nv --bind "$code:/pilot-code,/scratch:/scratch" "$image" \
 env PYTHONPATH=/pilot-code/src /workspace/.venv/bin/torchrun --standalone --nproc_per_node="${GPUS:-1}" \
 --module samudra.experiments.diffusion_global_train "${args[@]}" &
child=$!
set +e
wait "$child"
status=$?
if [[ "$requeue_requested" == 1 ]]; then
  wait "$child"
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
set -e
exit "$status"
