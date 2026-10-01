#!/bin/bash -l
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=latent-timing
#SBATCH --account=mit_general
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/orcd/pool/008/jrusak/diffusion-interior-engaging/logs/timing-%j.out
#SBATCH --error=/orcd/pool/008/jrusak/diffusion-interior-engaging/logs/timing-%j.err
set -euo pipefail
module load apptainer/1.5.2
: "${CODE_COMMIT:?Immutable benchmark producer required}"
root=/orcd/pool/008/jrusak/diffusion-interior-engaging
code="$root/code/$CODE_COMMIT"
image=/orcd/scratch/orcd/014/jrusak/apptainer_cache/images/diffusion-26.05-9cd36b1-amd64.sif
export TORCHINDUCTOR_COMPILE_THREADS=2
variants=()
if [[ "${COMPILE_TEST:-0}" == 1 ]]; then
  variants=(--variants baseline compiled_decoder skip_initial compiled_skip_initial)
fi
if [[ "${VALIDATION_COMPARISON:-0}" == 1 ]]; then
  variants=(--validation-comparison)
fi
export SAMUDRA_CODE_COMMIT="$CODE_COMMIT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
srun --gpus-per-task=1 apptainer exec --nv --bind "$code:/pilot-code,/orcd:/orcd" "$image" \
  env PYTHONPATH=/pilot-code/src /workspace/.venv/bin/python -m samudra.experiments.diffusion_benchmark \
  --root "$root" --checkpoint "$root/runs/latent-d192-extended-v1/D-1729/observation/best.pt" \
  --output "$root/runs/latent-performance-v1/job-$SLURM_JOB_ID" "${variants[@]}" \
  --precision "${BENCHMARK_PRECISION:-bf16}" --sample-index "${SAMPLE_INDEX:-0}"
