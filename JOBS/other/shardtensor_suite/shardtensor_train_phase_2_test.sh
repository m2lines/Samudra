#!/bin/bash
#SBATCH -p pi_abodner
#SBATCH --job-name=shardtensor-phase2-2x2-TEST4-1EPOCH
#SBATCH -N 1
#SBATCH --mem=300GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=16:00:00
#SBATCH -o /orcd/home/002/codycruz/Ocean_Emulator/logs/%x-%j.out
#SBATCH -e /orcd/home/002/codycruz/Ocean_Emulator/logs/%x-%j.out

set -euo pipefail

PROJECT_DIR="/orcd/home/002/codycruz/Ocean_Emulator"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
# A 320x320 cluster gives four 160x160 local tiles and a 10x10 deepest
# tile, large enough for the dilation-8 halo. Override all bounds together.
LLC_FACE="${LLC_FACE:-1}"
LLC_I_START="${LLC_I_START:-2880}"
LLC_I_END="${LLC_I_END:-3200}"
LLC_J_START="${LLC_J_START:-720}"
LLC_J_END="${LLC_J_END:-1040}"
EPOCHS="${EPOCHS:-1}"
DEBUG="${DEBUG:-true}"
DATA_NUM_WORKERS="${DATA_NUM_WORKERS:-6}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${SLURM_JOB_NAME:-shardtensor-phase2}-${SLURM_JOB_ID:-manual}}"

on_exit() {
  local exit_code=$?
  echo
  echo "======== Phase 2 ShardTensor 2x2 job finished (exit=${exit_code}) ========"
}
trap on_exit EXIT

module load miniforge/24.3.0-0
module load cuda/13.1.0

cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: expected Python environment at ${PYTHON_BIN}." >&2
  exit 1
fi

PROJECT_SITE_PACKAGES="${PROJECT_DIR}/.venv/lib/python3.11/site-packages"
export PYTHONPATH="${PROJECT_DIR}/src:${PROJECT_SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_FR_BUFFER_SIZE="${TORCH_FR_BUFFER_SIZE:-1048576}"

echo "======== ShardTensor Phase 2 2x2 curriculum smoke ========"
echo "started=$(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "job_id=${SLURM_JOB_ID:-<unset>} host=$(hostname) python=${PYTHON_BIN}"
echo "patch=face${LLC_FACE} i=[${LLC_I_START}:${LLC_I_END}) j=[${LLC_J_START}:${LLC_J_END})"
echo "mode=domain_parallel cluster_shape=[2,2] epochs=${EPOCHS} debug=${DEBUG} loss=mse"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
VISIBLE_GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if [[ "${VISIBLE_GPU_COUNT}" -lt 4 ]]; then
  echo "ERROR: the 2x2 ShardTensor job requires 4 visible GPUs; found ${VISIBLE_GPU_COUNT}." >&2
  exit 1
fi
"${PYTHON_BIN}" -c "import physicsnemo, torch; print('torch=' + torch.__version__); print('physicsnemo=' + getattr(physicsnemo, '__version__', 'unknown'))"

RUN_ARGS=(
  configs/samudra_llc/train.yaml
  --domain_parallel.cluster_shape "[2, 2]"
  --domain_parallel.use_fsdp false
  --domain_parallel.leader_scatter true
  --replay.enabled false
  --loss mse
  --epochs "${EPOCHS}"
  --debug "${DEBUG}"
  --preemptible false
  --batch_size 1
  --gradient_accumulation_steps 1
  --steps "[1]"
  --temporal_stride 1
  --data_stride "[1]"
  --model.pad constant
  --model.checkpointing null
  --model.use_bfloat16 false
  --model.corrector null
  --data.num_workers "${DATA_NUM_WORKERS}"
  --data.llc_face "${LLC_FACE}"
  --data.llc_i_start "${LLC_I_START}"
  --data.llc_i_end "${LLC_I_END}"
  --data.llc_j_start "${LLC_J_START}"
  --data.llc_j_end "${LLC_J_END}"
  --experiment.name "${EXPERIMENT_NAME}"
  --backend nccl
  --domain_parallel.enabled true
  --surface_snapshot true
)

LAUNCH=(
  "${PYTHON_BIN}" -m torch.distributed.run
  --standalone --nproc_per_node=4 --tee 3
  -m ocean_emulators.train
)

printf 'launch:'
printf ' %q' "${LAUNCH[@]}" "${RUN_ARGS[@]}"
printf '\n'

"${LAUNCH[@]}" "${RUN_ARGS[@]}"
