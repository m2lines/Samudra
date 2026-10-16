#!/bin/bash
#SBATCH --job-name=osn-large-chunk-io
#SBATCH --account=torch_pr_347_lzanna
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=8G
#SBATCH --time=00:20:00
set -euo pipefail
ROOT=/scratch/jr7309/benchmarks/osn-20261014
SIF=/scratch/jr7309/.apptainer-images/ghcr.io-m2lines-ocean-emulator-physicsnemo-26.05-23587a12baeb0f39e90a69c2560a5c6a5ddbed21.sif
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLOSC_NTHREADS=1
export APPTAINER_CACHEDIR=/scratch/jr7309/apptainer-cache
export SINGULARITY_CACHEDIR=/scratch/jr7309/singularity-cache
export APPTAINER_TMPDIR="${SLURM_TMPDIR:-/tmp}"
hostname
date -u +%FT%TZ
sha256sum "$ROOT/benchmark.py"
/share/apps/apptainer/bin/apptainer exec --bind /scratch:/scratch "$SIF" \
  /workspace/.venv/bin/python "$ROOT/benchmark.py" --start-concurrency "${START_CONCURRENCY:-1}" --skip-keys "${SKIP_KEYS:-0}" --out "$ROOT/results-$SLURM_JOB_ID"
