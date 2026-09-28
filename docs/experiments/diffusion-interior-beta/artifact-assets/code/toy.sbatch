#!/bin/bash -l
#SBATCH --job-name=diagonal-toy
#SBATCH --account=mit_general
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:25:00

#SBATCH --output=/orcd/pool/008/jrusak/diffusion-interior-engaging/logs/diagonal-toy-%j.out
#SBATCH --error=/orcd/pool/008/jrusak/diffusion-interior-engaging/logs/diagonal-toy-%j.err
set -euo pipefail
module load apptainer/1.5.2
root=/orcd/pool/008/jrusak/diffusion-interior-engaging
code="$root/code/af5218058dc0579712a6d980078cd79d0bf1a612"
# The completed evaluator snapshot has the same model implementation.
code="$root/code/f1f2be70fa4b0ce19b429408fa4020b8baa234c9"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export TOY_OUTPUT="$root/diagonal-toy-v1"
srun apptainer exec --nv --bind "$code:/pilot-code,/orcd:/orcd" /orcd/scratch/orcd/014/jrusak/apptainer_cache/images/diffusion-26.05-9cd36b1-amd64.sif env PYTHONPATH=/pilot-code/src /workspace/.venv/bin/python "$root/artifact-toy.py"
