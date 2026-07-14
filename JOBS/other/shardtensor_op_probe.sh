#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --job-name=shardtensor-op-probe-2D
#SBATCH -N 1
#SBATCH --mem=128GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH -o /orcd/home/002/codycruz/Ocean_Emulator/logs/%x-%j.out
#SBATCH -e /orcd/home/002/codycruz/Ocean_Emulator/logs/%x-%j.out

set -euo pipefail

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-<unset>}"
echo "SLURMD_NODENAME=${SLURMD_NODENAME:-<unset>}"
echo "hostname=$(hostname)"
echo "$(date '+%Y-%m-%d %H:%M:%S %Z')"

module load miniforge/24.3.0-0
module load cuda/13.1.0

PROJECT_DIR="/orcd/home/002/codycruz/Ocean_Emulator"
LOG_DIR="${PROJECT_DIR}/logs"

cd "${PROJECT_DIR}"

PROJECT_SITE_PACKAGES="${PROJECT_DIR}/.venv/lib/python3.11/site-packages"
export PYTHONPATH="${PROJECT_DIR}/src:${PROJECT_SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: expected Python environment at ${PYTHON_BIN}, but it is not executable." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

HEIGHT="${HEIGHT:-1088}"
WIDTH="${WIDTH:-1088}"
CHANNELS="${CHANNELS:-256}"
CONV_CHANNELS="${CONV_CHANNELS:-32}"
SMALL_SIDE="${SMALL_SIDE:-544}"
SMALL_CHANNELS="${SMALL_CHANNELS:-32}"

# All GPU-count runs append to this same report.
REPORT_NAME="${REPORT_NAME:-shardtensor-op-probe-${SLURM_JOB_ID:-manual}.out}"

echo "======== shardtensor op-support probe ========"
echo "height=${HEIGHT}, width=${WIDTH}, channels=${CHANNELS}"
echo "conv_channels=${CONV_CHANNELS}"
echo "small_side=${SMALL_SIDE}, small_channels=${SMALL_CHANNELS}"
echo "report=${LOG_DIR}/${REPORT_NAME}"

for GPUS in 1 2 3 4; do
  echo
  echo "======== running probe with ${GPUS} GPU(s) ========"

  EXTRA_ARGS=()
  if [[ "${GPUS}" -eq 1 ]]; then
    EXTRA_ARGS+=(--fresh)
  fi

  srun \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
    --gpus-per-task="${GPUS}" \
    "${PYTHON_BIN}" -m torch.distributed.run \
      --standalone \
      --nnodes=1 \
      --nproc_per_node="${GPUS}" \
      notebooks/shardtensor_op_probe.py \
      --height "${HEIGHT}" \
      --width "${WIDTH}" \
      --channels "${CHANNELS}" \
      --conv-channels "${CONV_CHANNELS}" \
      --small-side "${SMALL_SIDE}" \
      --small-channels "${SMALL_CHANNELS}" \
      --report-dir "${LOG_DIR}" \
      --report-name "${REPORT_NAME}" \
      "${EXTRA_ARGS[@]}"
done

echo
echo "======== all probe runs complete ========"
echo "report=${LOG_DIR}/${REPORT_NAME}"