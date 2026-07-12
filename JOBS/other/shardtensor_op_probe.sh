#!/bin/bash
#SBATCH -p pi_abodner
#SBATCH --job-name=shardtensor-op-probe
#SBATCH -N 1
#SBATCH --mem=128GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH -o /orcd/home/002/codycruz/Ocean_Emulator/logs/%x-%j.out
#SBATCH -e /orcd/home/002/codycruz/Ocean_Emulator/logs/%x-%j.out

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-<unset>}"
echo "SLURMD_NODENAME=${SLURMD_NODENAME:-<unset>}"
echo "hostname=$(hostname)"

module load miniforge/24.3.0-0
module load cuda/13.1.0

cd /orcd/home/002/codycruz/Ocean_Emulator

PROJECT_SITE_PACKAGES="/orcd/home/002/codycruz/Ocean_Emulator/.venv/lib/python3.11/site-packages"
export PYTHONPATH="/orcd/home/002/codycruz/Ocean_Emulator/src:${PROJECT_SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="${PYTHON_BIN:-/orcd/home/002/codycruz/Ocean_Emulator/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: expected Python 3.11 environment at ${PYTHON_BIN}, but it is not executable." >&2
  exit 1
fi

mkdir -p /orcd/home/002/codycruz/Ocean_Emulator/logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

if [[ -z "${GPUS:-}" ]]; then
  DETECTED_GPUS="$("${PYTHON_BIN}" -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)"
  if [[ "${DETECTED_GPUS}" -ge 2 ]]; then
    GPUS=2
  else
    GPUS=1
  fi
fi

HEIGHT="${HEIGHT:-1088}"
WIDTH="${WIDTH:-1088}"
CHANNELS="${CHANNELS:-256}"
CONV_CHANNELS="${CONV_CHANNELS:-32}"
SMALL_SIDE="${SMALL_SIDE:-544}"
SMALL_CHANNELS="${SMALL_CHANNELS:-32}"
SHARD_DIM="${SHARD_DIM:-2}"
REPORT_NAME="${REPORT_NAME:-shardtensor-op-probe-${SLURM_JOB_ID:-manual}.out}"

echo "======== shardtensor op-support probe ========"
echo "using GPUS=${GPUS}, height=${HEIGHT}, width=${WIDTH}, channels=${CHANNELS}"
echo "using conv_channels=${CONV_CHANNELS}, small_side=${SMALL_SIDE}, small_channels=${SMALL_CHANNELS}"
echo "using shard_dim=${SHARD_DIM}, report_name=${REPORT_NAME}"

"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node="${GPUS}" \
  notebooks/shardtensor_op_probe.py \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --channels "${CHANNELS}" \
  --conv-channels "${CONV_CHANNELS}" \
  --small-side "${SMALL_SIDE}" \
  --small-channels "${SMALL_CHANNELS}" \
  --shard-dim "${SHARD_DIM}" \
  --report-dir /orcd/home/002/codycruz/Ocean_Emulator/logs \
  --report-name "${REPORT_NAME}"
