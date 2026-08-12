#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

#SBATCH --account=torch_pr_347_lzanna
#SBATCH --partition=rtx6000_lzanna
#SBATCH --nodes=1
#SBATCH --ntasks=1
# The 2-degree jobs use two loader workers and stayed below 8 GiB of host RAM
# during the preflight. Keep the request small enough for Slurm backfill.
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=04:00:00

set -euo pipefail

: "${REPO_DIR:?Set REPO_DIR to the immutable experiment checkout}"
: "${CONFIG:?Set CONFIG relative to REPO_DIR}"
: "${NAME:?Set the unique output run name}"
: "${CODE_COMMIT:?Set CODE_COMMIT to the expected checkout commit}"

CURRENT_USER="$(id -un)"
PYTHON_BIN="${PYTHON_BIN:-/scratch/${CURRENT_USER}/samudra-venv/bin/python}"
OUTPUT_BASE="${OUTPUT_BASE:-/scratch/${CURRENT_USER}/runs/torch-2deg-perceiver-20260812}"
ACTUAL_COMMIT="$(git -C "${REPO_DIR}" rev-parse HEAD)"
if [[ "${ACTUAL_COMMIT}" != "${CODE_COMMIT}" ]]; then
  echo "Expected ${CODE_COMMIT}, found ${ACTUAL_COMMIT} in ${REPO_DIR}" >&2
  exit 1
fi
if [[ -n "$(git -C "${REPO_DIR}" status --porcelain --untracked-files=no)" ]]; then
  echo "Tracked files in ${REPO_DIR} are dirty" >&2
  exit 1
fi
if [[ -e "${OUTPUT_BASE}/${NAME}" ]]; then
  echo "Refusing to overwrite ${OUTPUT_BASE}/${NAME}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_BASE}" "/scratch/${CURRENT_USER}/.data_cache/${NAME}"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src"
export SAMUDRA_CODE_COMMIT="${CODE_COMMIT}"
export WANDB_GIT_COMMIT="${CODE_COMMIT}"
export WANDB_MODE=online
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# Torch may pack several single-GPU jobs onto one node. Avoid their process
# groups all attempting to bind Samudra's default port 40000.
export MASTER_PORT="$((10000 + SLURM_JOB_ID % 50000))"
export DATA_CACHE_DIR="/scratch/${CURRENT_USER}/.data_cache/${NAME}"
read -r -a extra_args <<< "${ARGS:-}"

echo "host=$(hostname) job=${SLURM_JOB_ID} commit=${CODE_COMMIT} config=${CONFIG}"
nvidia-smi
"${PYTHON_BIN}" - <<'PY'
import torch

print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print(f"device={torch.cuda.get_device_name(0)}")
print(f"sdpa_flash_enabled={torch.backends.cuda.flash_sdp_enabled()}")
PY

exec "${PYTHON_BIN}" -m samudra.train "${CONFIG}" \
  --experiment.name="${NAME}" \
  --experiment.base_output_dir="${OUTPUT_BASE}" \
  "${extra_args[@]}"
