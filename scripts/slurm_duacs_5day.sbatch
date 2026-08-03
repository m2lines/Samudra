#!/bin/bash

# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

# Slurm harness to run the DUACS daily->5-day-average reduction on Torch (NYU HPC)
# instead of a Coiled/AWS cluster. The job is CPU-only (a streaming dask reduction,
# no GPU, no xESMF) and reads/writes OSN directly -- both the source and the output
# live on OSN (nyu1.osn.mghpcc.org), which has no egress fees, so running the
# compute on HPC removes the Coiled/AWS bill entirely.
#
# Unlike the train/eval harnesses, this does NOT use Apptainer: the data subproject
# is developed against the `ocean_preprocessing` conda/mamba env (the PhysicsNeMo
# container does not carry s3fs/fire/xarray). Create it once on the cluster:
#   mamba env create -f data/mamba_env.yaml
#
# Submit with env vars (dry run first, then the real write):
#   DRY_RUN=1 sbatch scripts/slurm_duacs_5day.sbatch     # validate, write nothing
#   sbatch scripts/slurm_duacs_5day.sbatch               # full reduction
#
# Recognized env vars:
#   OSN_ENV_FILE   file sourced for OSN credentials (default: ~/.osn_env). Must export
#                  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (and optionally
#                  FSSPEC_S3_ENDPOINT_URL). Skipped if AWS_ACCESS_KEY_ID is already set.
#   ENV_NAME       conda/mamba env name (default: ocean_preprocessing)
#   PYTHON_BIN     explicit python to use, bypassing `mamba/conda run` env resolution
#   SRC            source Zarr path        (default: the script's built-in DEFAULT_SRC)
#   OUTPUT_PATH    destination Zarr path   (default: the script's built-in DEFAULT_OUTPUT)
#   WINDOW         coarsening window in days (default: 5)
#   DRY_RUN        if non-empty, pass --dry_run (prints structure + sample, no write)
#   ARGS           extra CLI overrides forwarded verbatim, e.g. ARGS="--small_run"

#SBATCH --job-name=oe-duacs-5day
#SBATCH --output=slurm-%j.out
#SBATCH --account=torch_pr_347_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
# Note: no --gres (CPU-only) and no --partition (let Slurm place the job).

set -euo pipefail

# --- OSN credentials -------------------------------------------------------
# Keep secrets out of the repo: source them from a file on the cluster. Skip if
# the credentials are already exported into the submission environment.
OSN_ENV_FILE="${OSN_ENV_FILE:-${HOME}/.osn_env}"
if [[ -z "${AWS_ACCESS_KEY_ID:-}" && -f "${OSN_ENV_FILE}" ]]; then
  echo "Sourcing OSN credentials from ${OSN_ENV_FILE}"
  # shellcheck disable=SC1090
  source "${OSN_ENV_FILE}"
fi
# OSN endpoint is not a secret; default it if unset.
export FSSPEC_S3_ENDPOINT_URL="${FSSPEC_S3_ENDPOINT_URL:-https://nyu1.osn.mghpcc.org/}"

if [[ -z "${AWS_ACCESS_KEY_ID:-}" || -z "${AWS_SECRET_ACCESS_KEY:-}" ]]; then
  echo "ERROR: OSN credentials not found." >&2
  echo "Export AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY, or place them in" >&2
  echo "${OSN_ENV_FILE} (a shell file that 'export's them). Tip: you can copy them" >&2
  echo "from your rclone config, e.g.:" >&2
  echo "  rclone config show nyu-osn | sed -n 's/^access_key_id *= *//p'" >&2
  exit 2
fi

# --- Python environment ----------------------------------------------------
# Pin numeric/blosc thread pools to single-threaded: the LocalCluster runs one
# process per worker, and blosc's multithreaded path is not safe under the
# concurrent Zarr chunk reads (matches the in-script defaults).
export BLOSC_NTHREADS=1 BLOSC_NOLOCK=1 NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Run a LocalCluster sized to the node's CPU allocation.
export OCEAN_DUACS_CLUSTER="${OCEAN_DUACS_CLUSTER:-local}"
export OCEAN_DUACS_WORKERS="${OCEAN_DUACS_WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"

ENV_NAME="${ENV_NAME:-ocean_preprocessing}"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_DIR}/data"

# Resolve a runner: explicit PYTHON_BIN wins; otherwise use `mamba/conda run -n`,
# which works in non-interactive batch shells without an activate hook.
if [[ -n "${PYTHON_BIN:-}" ]]; then
  RUNNER=("${PYTHON_BIN}")
elif command -v mamba >/dev/null 2>&1; then
  RUNNER=(mamba run -n "${ENV_NAME}" python)
elif command -v conda >/dev/null 2>&1; then
  RUNNER=(conda run --no-capture-output -n "${ENV_NAME}" python)
else
  echo "ERROR: no mamba/conda on PATH and PYTHON_BIN unset." >&2
  echo "Create the env with: mamba env create -f data/mamba_env.yaml" >&2
  echo "or set PYTHON_BIN=/path/to/envs/${ENV_NAME}/bin/python" >&2
  exit 3
fi

# --- Assemble CLI ----------------------------------------------------------
cli_args=()
[[ -n "${SRC:-}" ]] && cli_args+=("--src=${SRC}")
[[ -n "${OUTPUT_PATH:-}" ]] && cli_args+=("--output_path=${OUTPUT_PATH}")
[[ -n "${WINDOW:-}" ]] && cli_args+=("--window=${WINDOW}")
[[ -n "${DRY_RUN:-}" ]] && cli_args+=("--dry_run")
# shellcheck disable=SC2206
[[ -n "${ARGS:-}" ]] && cli_args+=(${ARGS})

echo "Repo dir:    ${REPO_DIR}/data"
echo "Env:         ${ENV_NAME}  (runner: ${RUNNER[*]})"
echo "Cluster:     ${OCEAN_DUACS_CLUSTER}  (workers: ${OCEAN_DUACS_WORKERS})"
echo "Endpoint:    ${FSSPEC_S3_ENDPOINT_URL}"
echo "Dry run:     ${DRY_RUN:+yes}${DRY_RUN:-no}"
echo "CLI:         ${cli_args[*]:-(defaults)}"

srun --ntasks=1 --ntasks-per-node=1 \
  "${RUNNER[@]}" -m ocean_preprocessing.make_duacs_5day "${cli_args[@]}"
