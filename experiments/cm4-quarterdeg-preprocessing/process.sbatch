#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=cm4-quarter-pilot
#SBATCH --account=torch_pr_347_lzanna
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
set -euo pipefail
export ROOT=/scratch/jr7309/cm4-quarterdeg-20261023
stage=${1:?pilot or full required}
export AWS_SHARED_CREDENTIALS_FILE=/scratch/jr7309/cm4-onedeg-20261015/private/aws-credentials
export AWS_PROFILE=cm4-source
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
export FSSPEC_S3_ENDPOINT_URL=https://nyu1.osn.mghpcc.org/
export AWS_REQUEST_CHECKSUM_CALCULATION=when_required AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
export PYTHONPATH="$ROOT/source/data:/scratch/jr7309/cm4-pr928-smoke-20261015/python_overlay"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export BLOSC_NTHREADS=1 BLOSC_NOLOCK=1 MPLBACKEND=Agg
export MPLCONFIGDIR="$ROOT/matplotlib" NUMBA_CACHE_DIR="$ROOT/numba-cache"
export DASK_TEMPORARY_DIRECTORY="${SLURM_TMPDIR:-/tmp}/cm4-${SLURM_JOB_ID}"
mkdir -p "$DASK_TEMPORARY_DIRECTORY"
cd "$ROOT"
sha256sum --check source.sha256
scontrol show job "$SLURM_JOB_ID" > "$ROOT/records/${stage}-allocation.txt"
on_exit() {
    rc=$?
    trap - EXIT
    python3 "$ROOT/private/send.py" --credential-file "$ROOT/private/${stage}-completion.json" || true
    exit "$rc"
}
trap on_exit EXIT
python3 "$ROOT/private/send.py" --credential-file "$ROOT/private/${stage}-completion.json" --check
srun --mpi=none --cpu-bind=cores /scratch/am16581/miniforge3/envs/ocean_preprocessing/bin/python -u "$ROOT/source/experiments/cm4-quarterdeg-preprocessing/run.py" "$stage"
