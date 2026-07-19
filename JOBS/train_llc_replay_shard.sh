#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --job-name=shardtensor-replay-2x2-DEBUG
#SBATCH -N 1
#SBATCH --mem=300GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -G h200:4
#SBATCH --time=02:00:00
#SBATCH -o /orcd/home/002/codycruz/Ocean_Emulator/logs/%x-%j.out
#SBATCH -e /orcd/home/002/codycruz/Ocean_Emulator/logs/%x-%j.out

set -euo pipefail

PROJECT_DIR="/orcd/home/002/codycruz/Ocean_Emulator"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"

# Keep the first replay gate small but UNet/halo-safe: each rank receives a
# 160x160 tile, whose deepest 10x10 tile is larger than the dilation-8 halo.
LLC_FACE="${LLC_FACE:-1}"
LLC_I_START="${LLC_I_START:-2880}"
LLC_I_END="${LLC_I_END:-3200}"
LLC_J_START="${LLC_J_START:-720}"
LLC_J_END="${LLC_J_END:-1040}"

DEBUG="${DEBUG:-true}"
EPOCHS="${EPOCHS:-1}"
DATA_NUM_WORKERS="${DATA_NUM_WORKERS:-4}"
DATA_PREFETCH_FACTOR="${DATA_PREFETCH_FACTOR:-2}"
REPLAY_BUFFER_SIZE="${REPLAY_BUFFER_SIZE:-8}"
REPLAY_STEPS_PER_EPOCH="${REPLAY_STEPS_PER_EPOCH:-64}"
REPLAY_REFRESH_EVERY="${REPLAY_REFRESH_EVERY:-4}"
REPLAY_MAX_LEAD_STEPS="${REPLAY_MAX_LEAD_STEPS:-[4]}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${SLURM_JOB_NAME:-shardtensor-replay}-${SLURM_JOB_ID:-manual}}"

on_exit() {
  local exit_code=$?
  echo
  echo "======== ShardTensor replay 2x2 job finished (exit=${exit_code}) ========"
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
export OCEAN_BLOSC_THREADS="${OCEAN_BLOSC_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_FR_BUFFER_SIZE="${TORCH_FR_BUFFER_SIZE:-1048576}"

echo "======== ShardTensor 2x2 temporal replay smoke ========"
echo "started=$(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "job_id=${SLURM_JOB_ID:-<unset>} host=$(hostname) python=${PYTHON_BIN}"
echo "patch=face${LLC_FACE} i=[${LLC_I_START}:${LLC_I_END}) j=[${LLC_J_START}:${LLC_J_END})"
echo "cluster_shape=[2,2] epochs=${EPOCHS} debug=${DEBUG} loss=mse"
echo "replay_buffer_size=${REPLAY_BUFFER_SIZE} replay_steps=${REPLAY_STEPS_PER_EPOCH} refresh_every=${REPLAY_REFRESH_EVERY} max_lead=${REPLAY_MAX_LEAD_STEPS}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

VISIBLE_GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if [[ "${VISIBLE_GPU_COUNT}" -ne 4 ]]; then
  echo "ERROR: the 2x2 replay job requires exactly 4 visible GPUs; found ${VISIBLE_GPU_COUNT}." >&2
  exit 1
fi

"${PYTHON_BIN}" -c "import physicsnemo, torch; from physicsnemo.domain_parallel import ShardTensor; print('torch=' + torch.__version__); print('physicsnemo=' + getattr(physicsnemo, '__version__', 'unknown')); print('ShardTensor OK')"

RUN_ARGS=(
  configs/samudra_llc/train_replay_shard.yaml
  --backend nccl
  --domain_parallel.enabled true
  --domain_parallel.cluster_shape "[2, 2]"
  --domain_parallel.use_fsdp false
  --domain_parallel.leader_scatter true
  --replay.enabled true
  --replay.buffer_size "${REPLAY_BUFFER_SIZE}"
  --replay.refresh_every_n_microbatches "${REPLAY_REFRESH_EVERY}"
  --replay.steps_per_epoch "${REPLAY_STEPS_PER_EPOCH}"
  --replay.max_lead_steps "${REPLAY_MAX_LEAD_STEPS}"
  --replay.checkpoint_buffer false
  --loss mse
  --epochs "${EPOCHS}"
  --debug "${DEBUG}"
  --preemptible false
  --emergency_checkpoint_interval_minutes 0
  --resume_ckpt_path null
  --batch_size 1
  --gradient_accumulation_steps 1
  --data_stride "[1]"
  --temporal_stride 1
  --steps "[1]"
  --model.pad constant
  --model.num_halo 0
  --model.num_sponge 0
  --model.pred_residuals false
  --model.checkpointing null
  --model.use_bfloat16 false
  --model.corrector null
  --surface_snapshot false
  --data.num_workers "${DATA_NUM_WORKERS}"
  --data.prefetch_factor "${DATA_PREFETCH_FACTOR}"
  --data.concurrent_compute false
  --pin_mem true
  --data.llc_face "${LLC_FACE}"
  --data.llc_i_start "${LLC_I_START}"
  --data.llc_i_end "${LLC_I_END}"
  --data.llc_j_start "${LLC_J_START}"
  --data.llc_j_end "${LLC_J_END}"
  --experiment.data_root /orcd/data/abodner/
  --experiment.name "${EXPERIMENT_NAME}"
)

LAUNCH=(
  "${PYTHON_BIN}" -m torch.distributed.run
  --standalone --nnodes=1 --nproc_per_node=4 --tee 3
  -m ocean_emulators.train
)

printf 'launch:'
printf ' %q' "${LAUNCH[@]}" "${RUN_ARGS[@]}"
printf '\n'

"${LAUNCH[@]}" "${RUN_ARGS[@]}"
