#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

# Submit the promoted full one-/half-degree S3 run and its checkpoint audits.
#
# Usage:
#   submit_coarse_latent_s3.sh \
#     <inverse-checkpoint> <physical-weight> <latent-weight> <code-layer> <container-sif>

set -euo pipefail

if (( $# != 5 )); then
  echo \
    "Usage: $0 <inverse-checkpoint> <physical-weight> <latent-weight> <code-layer> <container-sif>" \
    >&2
  exit 2
fi

INVERSE_CHECKPOINT="$(realpath "$1")"
PHYSICAL_WEIGHT="$2"
LATENT_WEIGHT="$3"
CODE_LAYER="$(realpath "$4")"
SIF_PATH="$(realpath "$5")"
CURRENT_USER="${USER:-$(id -un)}"
SCRATCH_DIR="${SCRATCH_DIR:-/scratch/${CURRENT_USER}}"
DATA_ROOT="${DATA_ROOT:-${SCRATCH_DIR}/data}"
OUTPUT_BASE="${OUTPUT_BASE:-${SCRATCH_DIR}/runs}"
LOG_DIR="${LOG_DIR:-${SCRATCH_DIR}/logs}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-${HOME}/slurm_apptainer_train.sbatch}"
AUDIT_SBATCH_SCRIPT="${AUDIT_SBATCH_SCRIPT:-${HOME}/slurm_apptainer_audit_coarse_dynamics.sbatch}"
AUDIT_SCRIPT="${AUDIT_SCRIPT:-${HOME}/audit_coarse_dynamics.py}"
INVERSE_AUDIT_SCRIPT="${INVERSE_AUDIT_SCRIPT:-${HOME}/audit_coarse_inverse.py}"
AUDIT_MAX_BATCHES="${AUDIT_MAX_BATCHES:-148}"
DATE_TAG="${DATE_TAG:-$(date +%Y-%m-%d)}"
OBJECTIVE_TAG="${OBJECTIVE_TAG:-wx${PHYSICAL_WEIGHT}-wz${LATENT_WEIGHT}}"
RUN_NAME="${RUN_NAME:-${DATE_TAG}-coarse-latent-s3-full-${OBJECTIVE_TAG}}"
VALIDATION_NAME="${RUN_NAME}-best-cross-validation"
WANDB_GROUP="${WANDB_GROUP:-coarse-latent-s3}"
TRAIN_NODES="${TRAIN_NODES:-1}"
TRAIN_GPUS_PER_NODE="${TRAIN_GPUS_PER_NODE:-8}"

for weight in "${PHYSICAL_WEIGHT}" "${LATENT_WEIGHT}"; do
  if [[ ! "${weight}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "Loss weights must be nonnegative decimal numbers; got ${weight}." >&2
    exit 2
  fi
done
if [[ "${PHYSICAL_WEIGHT}" == "0" && "${LATENT_WEIGHT}" == "0" ]]; then
  echo "At least one loss weight must be nonzero." >&2
  exit 2
fi
for count in "${TRAIN_NODES}" "${TRAIN_GPUS_PER_NODE}"; do
  if [[ ! "${count}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Training node/GPU counts must be positive integers; got ${count}." >&2
    exit 2
  fi
done
TRAIN_TOTAL_GPUS="$((TRAIN_NODES * TRAIN_GPUS_PER_NODE))"
if (( TRAIN_TOTAL_GPUS != 8 )); then
  echo \
    "S3 is calibrated for exactly eight workers; got ${TRAIN_NODES}x${TRAIN_GPUS_PER_NODE}=${TRAIN_TOTAL_GPUS}." \
    >&2
  exit 2
fi
TRAIN_CPUS_PER_TASK="$((8 * TRAIN_GPUS_PER_NODE))"
TRAIN_MEMORY="$((64 * TRAIN_GPUS_PER_NODE))G"
if [[ ! "${AUDIT_MAX_BATCHES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "AUDIT_MAX_BATCHES must be a positive integer." >&2
  exit 2
fi

for required_file in \
  "${INVERSE_CHECKPOINT}" \
  "${CODE_LAYER}" \
  "${CODE_LAYER}.sha256" \
  "${CODE_LAYER}.json" \
  "${SIF_PATH}" \
  "${SBATCH_SCRIPT}" \
  "${AUDIT_SBATCH_SCRIPT}" \
  "${AUDIT_SCRIPT}" \
  "${INVERSE_AUDIT_SCRIPT}"; do
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
for run_directory in \
  "${OUTPUT_BASE}/${RUN_NAME}" \
  "${OUTPUT_BASE}/${VALIDATION_NAME}"; do
  if [[ -e "${run_directory}" ]]; then
    echo "Refusing to reuse existing run directory: ${run_directory}" >&2
    exit 5
  fi
done
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY must be set so route/depth metrics are retained." >&2
  exit 4
fi

TRAIN_CONFIG="configs/samudra_multi_om4/train_cross_1_halfdeg_coarse_latent_dynamics_full.yaml"
VALIDATION_CONFIG="configs/samudra_multi_om4/validate_cross_1_halfdeg_coarse_latent_dynamics.yaml"
run_directory="${OUTPUT_BASE}/${RUN_NAME}"
best_checkpoint="${run_directory}/saved_nets/best_validation_ckpt.pt"
train_args=(
  "--resume_ckpt_path=${INVERSE_CHECKPOINT}"
  "--model.physical_forecast_loss_weight=${PHYSICAL_WEIGHT}"
  "--model.latent_teacher_loss_weight=${LATENT_WEIGHT}"
  "--experiment.wandb.group=${WANDB_GROUP}"
  "--preemptible=true"
)
train_args_string="${train_args[*]}"

train_job_id="$(
  CONFIG="${TRAIN_CONFIG}" \
  NAME="${RUN_NAME}" \
  PYTHON_MODULE="samudra.train" \
  ARGS="${train_args_string}" \
  DATA_ROOT="${DATA_ROOT}" \
  OUTPUT_BASE="${OUTPUT_BASE}" \
  SCRATCH_DIR="${SCRATCH_DIR}" \
  SIF_PATH="${SIF_PATH}" \
  CODE_LAYER="${CODE_LAYER}" \
  WANDB_MODE="online" \
  GPUS_PER_NODE="${TRAIN_GPUS_PER_NODE}" \
  REQUEUE_ON_USR1="1" \
  DATA_CACHE_DIR="${SCRATCH_DIR}/.data_cache/${RUN_NAME}" \
    sbatch \
      --parsable \
      --nodes="${TRAIN_NODES}" \
      --ntasks-per-node="1" \
      --chdir="${SCRATCH_DIR}" \
      --job-name="oe-s3-full" \
      --account="torch_pr_347_courant" \
      --constraint="h200" \
      --gres="gpu:${TRAIN_GPUS_PER_NODE}" \
      --cpus-per-task="${TRAIN_CPUS_PER_TASK}" \
      --mem="${TRAIN_MEMORY}" \
      --time="12:00:00" \
      --signal="B:USR1@120" \
      --requeue \
      --comment="preemption=yes;preemption_partitions_only=yes;requeue=true" \
      --output="${LOG_DIR}/${RUN_NAME}-%j.out" \
      --error="${LOG_DIR}/${RUN_NAME}-%j.err" \
      "${SBATCH_SCRIPT}"
)"

validation_args=(
  "--resume_ckpt_path=${best_checkpoint}"
  "--model.physical_forecast_loss_weight=${PHYSICAL_WEIGHT}"
  "--model.latent_teacher_loss_weight=${LATENT_WEIGHT}"
  "--experiment.wandb.group=${WANDB_GROUP}-cross-validation"
  "--preemptible=true"
)
validation_args_string="${validation_args[*]}"
validation_job_id="$(
  CONFIG="${VALIDATION_CONFIG}" \
  NAME="${VALIDATION_NAME}" \
  PYTHON_MODULE="samudra.train" \
  ARGS="${validation_args_string}" \
  DATA_ROOT="${DATA_ROOT}" \
  OUTPUT_BASE="${OUTPUT_BASE}" \
  SCRATCH_DIR="${SCRATCH_DIR}" \
  SIF_PATH="${SIF_PATH}" \
  CODE_LAYER="${CODE_LAYER}" \
  WANDB_MODE="online" \
  GPUS_PER_NODE="1" \
  REQUEUE_ON_USR1="1" \
  DATA_CACHE_DIR="${SCRATCH_DIR}/.data_cache/${VALIDATION_NAME}" \
    sbatch \
      --parsable \
      --dependency="afterok:${train_job_id}" \
      --chdir="${SCRATCH_DIR}" \
      --job-name="oe-s3-val" \
      --account="torch_pr_347_courant" \
      --constraint="h200" \
      --gres="gpu:1" \
      --cpus-per-task="8" \
      --mem="64G" \
      --time="01:00:00" \
      --signal="B:USR1@120" \
      --requeue \
      --comment="preemption=yes;preemption_partitions_only=yes;requeue=true" \
      --output="${LOG_DIR}/${VALIDATION_NAME}-%j.out" \
      --error="${LOG_DIR}/${VALIDATION_NAME}-%j.err" \
      "${SBATCH_SCRIPT}"
)"

audit_job_id="$(
  RUN_DIR="${run_directory}" \
  CHECKPOINT="${best_checkpoint}" \
  INVERSE_CHECKPOINT="${INVERSE_CHECKPOINT}" \
  DATA_ROOT="${DATA_ROOT}" \
  SIF_PATH="${SIF_PATH}" \
  CODE_LAYER="${CODE_LAYER}" \
  AUDIT_SCRIPT="${AUDIT_SCRIPT}" \
  INVERSE_AUDIT_SCRIPT="${INVERSE_AUDIT_SCRIPT}" \
  MAX_BATCHES="${AUDIT_MAX_BATCHES}" \
    sbatch \
      --parsable \
      --dependency="afterok:${train_job_id}" \
      --chdir="${SCRATCH_DIR}" \
      --job-name="oe-s3-audit" \
      --account="torch_pr_347_courant" \
      --constraint="h200" \
      --gres="gpu:1" \
      --cpus-per-task="8" \
      --mem="64G" \
      --time="00:30:00" \
      --requeue \
      --comment="preemption=yes;preemption_partitions_only=yes;requeue=true" \
      --output="${LOG_DIR}/${RUN_NAME}-audit-%j.out" \
      --error="${LOG_DIR}/${RUN_NAME}-audit-%j.err" \
      "${AUDIT_SBATCH_SCRIPT}"
)"

printf \
  'train_job_id\tvalidation_job_id\taudit_job_id\trun_name\tphysical_weight\tlatent_weight\tlayout\n'
printf '%s\t%s\t%s\t%s\t%s\t%s\t%sx%s\n' \
  "${train_job_id}" \
  "${validation_job_id}" \
  "${audit_job_id}" \
  "${RUN_NAME}" \
  "${PHYSICAL_WEIGHT}" \
  "${LATENT_WEIGHT}" \
  "${TRAIN_NODES}" \
  "${TRAIN_GPUS_PER_NODE}"
