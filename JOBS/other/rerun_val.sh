#!/bin/bash
#SBATCH -p pi_abodner
#SBATCH --job-name=2026-04-22-Samudra_LLC:rerun_val_experiment_2
#SBATCH -N 1
#SBATCH --mem=500GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --gres=gpu:4
#SBATCH --time=00-4:00:00
#SBATCH -o /home/codycruz/Ocean_Emulator/logs/%x-%j.out
#SBATCH -e /home/codycruz/Ocean_Emulator/logs/%x-%j.out

set -euo pipefail

# Keep this explicit: under Slurm, ${BASH_SOURCE[0]} may refer to the staged
# batch script path or a relative submission path, which makes repo-root
# inference unreliable.
REPO_DIR="/orcd/home/002/codycruz/Ocean_Emulator"

# KNOBS
GPUS="${GPUS:-4}"
CONFIG_PATH="${CONFIG_PATH:-/home/codycruz/Ocean_Emulator/.LOCAL/2026-04-21-Samudra_LLC:config_tests_experiment_2/config.yaml}"
CKPT_PATH="${CKPT_PATH:-/home/codycruz/Ocean_Emulator/.LOCAL/2026-04-21-Samudra_LLC:config_tests_experiment_2/saved_nets/best_validation_ckpt.pt}"
OUT_PATH="${OUT_PATH:-/home/codycruz/Ocean_Emulator/.LOCAL/2026-04-22-rerun-val:config_tests_experiment_2}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"
WANDB_MODE="${WANDB_MODE:-offline}"
BACKEND="${BACKEND:-nccl}"
DATA_NUM_WORKERS="${DATA_NUM_WORKERS:-2}"
DDP_TIMEOUT_MINUTES="${DDP_TIMEOUT_MINUTES:-300}"
DRY_RUN="${DRY_RUN:-false}"

if [[ -z "${CONFIG_PATH}" ]]; then
  echo "ERROR: CONFIG_PATH must be set." >&2
  exit 1
fi

if [[ -z "${CKPT_PATH}" ]]; then
  echo "ERROR: CKPT_PATH must be set." >&2
  exit 1
fi

if [[ -n "${OUT_PATH}" ]]; then
  if [[ -n "${BASE_OUTPUT_DIR}" || -n "${EXPERIMENT_NAME}" ]]; then
    echo "ERROR: OUT_PATH is mutually exclusive with BASE_OUTPUT_DIR/EXPERIMENT_NAME." >&2
    exit 1
  fi
  BASE_OUTPUT_DIR="$(dirname "${OUT_PATH}")"
  EXPERIMENT_NAME="$(basename "${OUT_PATH}")"
fi

if [[ -z "${BASE_OUTPUT_DIR}" ]]; then
  BASE_OUTPUT_DIR="${REPO_DIR}/.LOCAL"
fi

if [[ -z "${EXPERIMENT_NAME}" ]]; then
  EXPERIMENT_NAME="rerun-val-$(date +%Y-%m-%d-%H%M%S)"
fi
EXPERIMENT_NAME="${EXPERIMENT_NAME}${SLURM_JOB_ID:+-${SLURM_JOB_ID}}"

if [[ "${GPUS}" -lt 1 ]]; then
  echo "ERROR: GPUS must be >= 1 (got ${GPUS})." >&2
  exit 1
fi

if [[ "${WANDB_MODE}" != "offline" && "${WANDB_MODE}" != "online" && "${WANDB_MODE}" != "disabled" ]]; then
  echo "ERROR: WANDB_MODE must be one of offline, online, disabled (got ${WANDB_MODE})." >&2
  exit 1
fi

if [[ -n "${SLURM_GPUS_ON_NODE:-}" && "${SLURM_GPUS_ON_NODE}" != "${GPUS}" ]]; then
  echo "WARNING: SLURM allocated ${SLURM_GPUS_ON_NODE} GPUs but GPUS=${GPUS}." >&2
  echo "         Keep these aligned, e.g. sbatch --gres=gpu:${GPUS} JOBS/rerun_val.sh" >&2
fi

CMD=(
  uv run python -m torch.distributed.run
  --standalone
  --nnodes=1
  --nproc_per_node="${GPUS}"
  -m ocean_emulators.validate_checkpoint
  "${CONFIG_PATH}"
  --backend "${BACKEND}"
  --resume_ckpt_path "${CKPT_PATH}"
  --experiment.name "${EXPERIMENT_NAME}"
  --experiment.base_output_dir "${BASE_OUTPUT_DIR}"
  --experiment.wandb.mode "${WANDB_MODE}"
  --data.num_workers "${DATA_NUM_WORKERS}"
  --ddp_timeout_minutes "${DDP_TIMEOUT_MINUTES}"
)

echo "======== rerun one-step validation from checkpoint ========"
echo "config: ${CONFIG_PATH}"
echo "checkpoint: ${CKPT_PATH}"
echo "output dir: ${BASE_OUTPUT_DIR}/${EXPERIMENT_NAME}"
echo "wandb mode: ${WANDB_MODE}"
echo "backend: ${BACKEND}"
echo "gpus: ${GPUS}"
echo "data workers: ${DATA_NUM_WORKERS}"
echo "repo dir: ${REPO_DIR}"
if [[ "${GPUS}" -gt 1 && "${BACKEND}" == "cuda" ]]; then
  echo "ERROR: BACKEND=cuda with GPUS>1 launches multiple non-distributed processes on GPU 0." >&2
  echo "       Use BACKEND=nccl or BACKEND=auto for multi-GPU reruns." >&2
  exit 1
fi

if [[ "${DRY_RUN}" == "true" ]]; then
  printf 'Command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

module load miniforge/24.3.0-0
module load cuda/13.1.0

cd "${REPO_DIR}"
if [[ ! -f pyproject.toml ]]; then
  echo "ERROR: Expected pyproject.toml in ${REPO_DIR}, but it was not found." >&2
  exit 1
fi
uv sync --dev

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export NCCL_DEBUG=INFO

"${CMD[@]}"
