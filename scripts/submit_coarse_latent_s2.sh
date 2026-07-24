#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

# Submit the four matched half-degree S2 objective arms on Torch.
#
# Usage:
#   submit_coarse_latent_s2.sh <inverse-checkpoint> <code-layer> <container-sif>

set -euo pipefail

if (( $# != 3 )); then
  echo "Usage: $0 <inverse-checkpoint> <code-layer> <container-sif>" >&2
  exit 2
fi

INVERSE_CHECKPOINT="$(realpath "$1")"
CODE_LAYER="$(realpath "$2")"
SIF_PATH="$(realpath "$3")"
CURRENT_USER="${USER:-$(id -un)}"
SCRATCH_DIR="${SCRATCH_DIR:-/scratch/${CURRENT_USER}}"
DATA_ROOT="${DATA_ROOT:-${SCRATCH_DIR}/data}"
OUTPUT_BASE="${OUTPUT_BASE:-${SCRATCH_DIR}/runs}"
LOG_DIR="${LOG_DIR:-${SCRATCH_DIR}/logs}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-${HOME}/slurm_apptainer_train.sbatch}"
DATE_TAG="${DATE_TAG:-$(date +%Y-%m-%d)}"
RUN_PREFIX="${RUN_PREFIX:-${DATE_TAG}-coarse-latent-s2}"
WANDB_GROUP="${WANDB_GROUP:-coarse-latent-s2}"

for required_file in \
  "${INVERSE_CHECKPOINT}" \
  "${CODE_LAYER}" \
  "${CODE_LAYER}.sha256" \
  "${CODE_LAYER}.json" \
  "${SIF_PATH}" \
  "${SBATCH_SCRIPT}"; do
  if [[ ! -s "${required_file}" ]]; then
    echo "Required input is missing or empty: ${required_file}" >&2
    exit 3
  fi
done
for required_directory in "${DATA_ROOT}" "${OUTPUT_BASE}" "${LOG_DIR}"; do
  if [[ ! -d "${required_directory}" ]]; then
    echo "Required directory does not exist: ${required_directory}" >&2
    exit 3
  fi
done
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY must be set so route/depth metrics are retained." >&2
  exit 4
fi

CONFIG="configs/samudra_multi_om4/train_halfdeg_coarse_latent_dynamics_proxy.yaml"
arms=(
  "physical-only:1:0"
  "latent-only:0:1"
  "combined-001:1:0.01"
  "combined-01:1:0.1"
)

printf 'arm\tjob_id\trun_name\tphysical_weight\tlatent_weight\n'
for specification in "${arms[@]}"; do
  IFS=: read -r arm physical_weight latent_weight <<<"${specification}"
  name="${RUN_PREFIX}-${arm}"
  run_directory="${OUTPUT_BASE}/${name}"
  if [[ -e "${run_directory}" ]]; then
    echo "Refusing to reuse existing run directory: ${run_directory}" >&2
    exit 5
  fi

  args=(
    "--resume_ckpt_path=${INVERSE_CHECKPOINT}"
    "--model.physical_forecast_loss_weight=${physical_weight}"
    "--model.latent_teacher_loss_weight=${latent_weight}"
    "--experiment.wandb.group=${WANDB_GROUP}"
    "--preemptible=true"
  )
  args_string="${args[*]}"

  job_id="$(
    CONFIG="${CONFIG}" \
    NAME="${name}" \
    PYTHON_MODULE="samudra.train" \
    ARGS="${args_string}" \
    DATA_ROOT="${DATA_ROOT}" \
    OUTPUT_BASE="${OUTPUT_BASE}" \
    SCRATCH_DIR="${SCRATCH_DIR}" \
    SIF_PATH="${SIF_PATH}" \
    CODE_LAYER="${CODE_LAYER}" \
    WANDB_MODE="online" \
    GPUS_PER_NODE="1" \
    REQUEUE_ON_USR1="1" \
    DATA_CACHE_DIR="${SCRATCH_DIR}/.data_cache/${name}" \
      sbatch \
        --parsable \
        --chdir="${SCRATCH_DIR}" \
        --job-name="oe-s2-${arm}" \
        --account="torch_pr_347_courant" \
        --constraint="h200" \
        --gres="gpu:1" \
        --cpus-per-task="8" \
        --mem="64G" \
        --time="04:00:00" \
        --signal="B:USR1@120" \
        --requeue \
        --comment="preemption=yes;preemption_partitions_only=yes;requeue=true" \
        --output="${LOG_DIR}/${name}-%j.out" \
        --error="${LOG_DIR}/${name}-%j.err" \
        "${SBATCH_SCRIPT}"
  )"
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${arm}" "${job_id}" "${name}" "${physical_weight}" "${latent_weight}"
done
