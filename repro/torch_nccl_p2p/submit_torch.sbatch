#!/bin/bash
# Minimal single-node NCCL repro for NYU Torch RTX6000 nodes.
#
# Submit without the workaround:
#   sbatch --account=<your-account> repro/torch_nccl_p2p/submit_torch.sbatch
#
# Submit with the documented workaround:
#   USE_P2P_FIX=1 sbatch --account=<your-account> repro/torch_nccl_p2p/submit_torch.sbatch
#
# If you created a local venv from pyproject.toml, point PYTHON_BIN at it:
#   PYTHON_BIN=$PWD/repro/torch_nccl_p2p/.venv/bin/python sbatch --account=<acct> ...

#SBATCH --job-name=nccl-p2p-repro
#SBATCH --output=slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=1400G
#SBATCH --gres=gpu:rtx6000:8
#SBATCH --time=00:15:00

set -euo pipefail

# Under sbatch, BASH_SOURCE points at Slurm's private spool copy of the script.
# Use the submit directory by default so we can find sibling files and write traces.
SCRIPT_DIR="$(cd -- "${REPRO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-120}"
USE_P2P_FIX="${USE_P2P_FIX:-0}"

if [[ "${USE_P2P_FIX}" == "1" ]]; then
  export NCCL_P2P_DISABLE=1
  export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
fi

export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,COLL}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
export TORCH_FR_BUFFER_SIZE="${TORCH_FR_BUFFER_SIZE:-1048576}"
export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-1}"
export TIMEOUT_SECONDS

TRACE_DIR="${TRACE_DIR:-${SCRIPT_DIR}/trace-${SLURM_JOB_ID:-manual}}"
mkdir -p "${TRACE_DIR}"
export TORCH_FR_DUMP_TEMP_FILE="${TORCH_FR_DUMP_TEMP_FILE:-${TRACE_DIR}/comm_trace_rank_}"
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="${TORCH_NCCL_DEBUG_INFO_TEMP_FILE:-${TRACE_DIR}/comm_trace_rank_}"

GPUS_PER_NODE="${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-}}"
if [[ -z "${GPUS_PER_NODE}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS_PER_NODE="$(nvidia-smi -L | wc -l | tr -d ' ')"
  else
    echo "Could not determine GPUS_PER_NODE." >&2
    exit 1
  fi
fi

echo "host:                 $(hostname)"
echo "python:               ${PYTHON_BIN}"
echo "gpus_per_node:        ${GPUS_PER_NODE}"
echo "timeout_seconds:      ${TIMEOUT_SECONDS}"
echo "use_p2p_fix:          ${USE_P2P_FIX}"
echo "NCCL_P2P_DISABLE:     ${NCCL_P2P_DISABLE:-unset}"
echo "TORCH_NCCL_ASYNC_ERROR_HANDLING: ${TORCH_NCCL_ASYNC_ERROR_HANDLING:-unset}"
echo "trace_dir:            ${TRACE_DIR}"

"${PYTHON_BIN}" -c "import torch; print(f'torch={torch.__version__} cuda={torch.version.cuda} cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}')"

exec "${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${GPUS_PER_NODE}" \
  "${SCRIPT_DIR}/nccl_barrier_repro.py"
