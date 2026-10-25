#!/bin/bash
#SBATCH --job-name=hfds-origins
#SBATCH --account=torch_pr_347_lzanna
#SBATCH --partition=rtx6000_lzanna
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
set -euo pipefail
ROOT=/scratch/jr7309/om4-hfds-ablation-20261021
OLD=/orcd/pool/008/jrusak/om4-hfds-ablation-20261021
cd "$ROOT"
test -f TRANSFER-VERIFIED
export PYTHONPATH="$ROOT/code/src" PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export APPTAINERENV_PYTHONNOUSERSITE=1 MPLBACKEND=Agg APPTAINERENV_MPLBACKEND=Agg
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2
export APPTAINER_CACHEDIR=/scratch/jr7309/apptainer-cache
export APPTAINER_TMPDIR="${SLURM_TMPDIR:-/tmp}"
export DATA_CACHE_DIR="$ROOT/cache/data"
SIF=/scratch/jr7309/.apptainer-images/ghcr.io-m2lines-ocean-emulator-physicsnemo-26.05-23587a12baeb0f39e90a69c2560a5c6a5ddbed21.sif

apptainer exec --nv --bind /scratch:/scratch --bind "$ROOT:$OLD" --bind /scratch/jr7309/data/om4_onedeg_v3:/orcd/data/abodner/002/jrusak/om4_onedeg_v3:ro --bind "$ROOT/normalization:/orcd/pool/008/jrusak/om4-current-structure-20261020/normalization:ro" --bind "$ROOT/runtime-fixes/plotting.py:/workspace/.venv/lib/python3.12/site-packages/xarrayutils/plotting.py:ro" "$SIF" /workspace/.venv/bin/python -u "$ROOT/multi_origin.py" "$SLURM_ARRAY_TASK_ID"
