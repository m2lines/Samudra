#!/bin/bash
#SBATCH --job-name=om4-structure-scales
#SBATCH --account=torch_pr_347_lzanna
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail
: "${ROOT:?}" "${DATA_ROOT:?}" "${SIF:?}"
export PYTHONPATH="$ROOT/code/src"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export APPTAINER_TMPDIR="${SLURM_TMPDIR:-/tmp}"
cd "$ROOT"
sha256sum --check --quiet manifest.sha256
apptainer exec --bind /scratch:/scratch "$SIF" /workspace/.venv/bin/python -u "$ROOT/code/experiments/structure-loss/prepare_scales.py" --data "$DATA_ROOT/OM4.zarr" --output "$ROOT/scales"
apptainer exec --bind /scratch:/scratch "$SIF" /workspace/.venv/bin/python -u "$ROOT/code/experiments/structure-loss/make_configs.py" --root "$ROOT" --data-root "$DATA_ROOT"
touch "$ROOT/PREPARED"
