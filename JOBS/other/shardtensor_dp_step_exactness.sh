#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --job-name=shardtensor-dp-exactness-test1
#SBATCH -x node4100,node3401,node3000
#SBATCH -N 1
#SBATCH --mem=128GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -G h200:4
#SBATCH --time=02:00:00
#SBATCH -o /orcd/home/002/codycruz/Ocean_Emulator/logs/%x-%j.out
#SBATCH -e /orcd/home/002/codycruz/Ocean_Emulator/logs/%x-%j.out

set -euo pipefail

PROJECT_DIR="/orcd/home/002/codycruz/Ocean_Emulator"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"

HEIGHT="${HEIGHT:-1088}"
WIDTH="${WIDTH:-1088}"
BATCH_SIZE="${BATCH_SIZE:-1}"
IN_CHANNELS="${IN_CHANNELS:-32}"
OUT_CHANNELS="${OUT_CHANNELS:-8}"
WIDTHS="${WIDTHS:-32 48 64 64}"
UPSCALE_FACTOR="${UPSCALE_FACTOR:-2}"
SEED="${SEED:-20260715}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
ATOL="${ATOL:-0.0002}"
RTOL="${RTOL:-0.0002}"
MAX_RELATIVE_L2="${MAX_RELATIVE_L2:-0.002}"
POST_STEP_ATOL="${POST_STEP_ATOL:-0.0005}"
POST_STEP_RTOL="${POST_STEP_RTOL:-0.0002}"
POST_STEP_MAX_RELATIVE_L2="${POST_STEP_MAX_RELATIVE_L2:-0.0002}"

on_exit() {
  local exit_code=$?
  echo
  echo "======== ShardTensor 2x2 exactness job finished (exit=${exit_code}) ========"
  echo "log=${PROJECT_DIR}/logs/${SLURM_JOB_NAME:-shardtensor-dp-exactness}-${SLURM_JOB_ID:-manual}.out"
}
trap on_exit EXIT

module load miniforge/24.3.0-0
module load cuda/13.1.0

cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: expected Python environment at ${PYTHON_BIN}, but it is not executable." >&2
  exit 1
fi

read -r -a WIDTH_ARGS <<< "${WIDTHS}"
if [[ "${#WIDTH_ARGS[@]}" -ne 4 ]]; then
  echo "ERROR: WIDTHS must contain exactly four U-Net widths (got '${WIDTHS}')." >&2
  exit 1
fi

PROJECT_SITE_PACKAGES="${PROJECT_DIR}/.venv/lib/python3.11/site-packages"
export PYTHONPATH="${PROJECT_DIR}/src:${PROJECT_SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=120

echo "======== ShardTensor 2x2 Samudra optimizer-step exactness ========"
echo "started=$(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "job_id=${SLURM_JOB_ID:-<unset>} node=${SLURMD_NODENAME:-<unset>} host=$(hostname)"
echo "python=${PYTHON_BIN}"
echo "shape=(batch=${BATCH_SIZE}, in=${IN_CHANNELS}, height=${HEIGHT}, width=${WIDTH}, out=${OUT_CHANNELS})"
echo "unet_widths=${WIDTHS} upscale_factor=${UPSCALE_FACTOR} seed=${SEED} lr=${LEARNING_RATE}"
echo "tolerance=(atol=${ATOL}, rtol=${RTOL}, max_relative_l2=${MAX_RELATIVE_L2})"
echo "post_step_tolerance=(atol=${POST_STEP_ATOL}, rtol=${POST_STEP_RTOL}, max_relative_l2=${POST_STEP_MAX_RELATIVE_L2})"
echo "cuda_module=$(module list 2>&1 | tr '\n' ' ')"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
"${PYTHON_BIN}" -c "import physicsnemo, torch; print('torch=' + torch.__version__); print('physicsnemo=' + getattr(physicsnemo, '__version__', 'unknown')); print('cuda=' + torch.version.cuda); print('visible_gpus=' + str(torch.cuda.device_count()))"

RUN_ARGS=(
  --height "${HEIGHT}"
  --width "${WIDTH}"
  --batch-size "${BATCH_SIZE}"
  --in-channels "${IN_CHANNELS}"
  --out-channels "${OUT_CHANNELS}"
  --widths "${WIDTH_ARGS[@]}"
  --upscale-factor "${UPSCALE_FACTOR}"
  --seed "${SEED}"
  --lr "${LEARNING_RATE}"
  --atol "${ATOL}"
  --rtol "${RTOL}"
  --max-relative-l2 "${MAX_RELATIVE_L2}"
  --post-step-atol "${POST_STEP_ATOL}"
  --post-step-rtol "${POST_STEP_RTOL}"
  --post-step-max-relative-l2 "${POST_STEP_MAX_RELATIVE_L2}"
)

printf 'launch:'
printf ' %q' "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=4 notebooks/dp_step_exactness.py "${RUN_ARGS[@]}"
printf '\n'

"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  --tee 3 \
  notebooks/dp_step_exactness.py \
  "${RUN_ARGS[@]}"
