#!/bin/bash -l
#SBATCH --job-name=diagonal-modes
#SBATCH --account=mit_general
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --array=1
#SBATCH --output=/orcd/pool/008/jrusak/diffusion-interior-engaging/logs/diagonal-modes-%A_%a.out
#SBATCH --error=/orcd/pool/008/jrusak/diffusion-interior-engaging/logs/diagonal-modes-%A_%a.err
set -euo pipefail
module load apptainer/1.5.2
root=/orcd/pool/008/jrusak/diffusion-interior-engaging
code="$root/code/af5218058dc0579712a6d980078cd79d0bf1a612"
# The completed evaluator snapshot has the same model implementation.
code="$root/code/f1f2be70fa4b0ce19b429408fa4020b8baa234c9"
seed=$((1729 + SLURM_ARRAY_TASK_ID / 2))
phase=om4
if (( SLURM_ARRAY_TASK_ID % 2 )); then phase=observation; fi
export SAMUDRA_CODE_COMMIT=f1f2be70fa4b0ce19b429408fa4020b8baa234c9
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
srun apptainer exec --nv --bind "$code:/pilot-code,/orcd:/orcd" /orcd/scratch/orcd/014/jrusak/apptainer_cache/images/diffusion-26.05-9cd36b1-amd64.sif env PYTHONPATH=/pilot-code/src /workspace/.venv/bin/python "$root/artifact-modes-probe.py" --root "$root" --checkpoint "$root/runs/scratch-b192-v2/B-$seed/$phase/best.pt" --output "$root/diagonal-modes-v1/B-$seed/$phase"
