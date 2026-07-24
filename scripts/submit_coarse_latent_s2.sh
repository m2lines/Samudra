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
AUDIT_SBATCH_SCRIPT="${AUDIT_SBATCH_SCRIPT:-${HOME}/slurm_apptainer_audit_coarse_dynamics.sbatch}"
AUDIT_SCRIPT="${AUDIT_SCRIPT:-${HOME}/audit_coarse_dynamics.py}"
INVERSE_AUDIT_SCRIPT="${INVERSE_AUDIT_SCRIPT:-${HOME}/audit_coarse_inverse.py}"
AUDIT_MAX_BATCHES="${AUDIT_MAX_BATCHES:-148}"
DATE_TAG="${DATE_TAG:-$(date +%Y-%m-%d)}"
RUN_PREFIX="${RUN_PREFIX:-${DATE_TAG}-coarse-latent-s2}"
WANDB_GROUP="${WANDB_GROUP:-coarse-latent-s2}"

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
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY must be set so route/depth metrics are retained." >&2
  exit 4
fi
if [[ ! "${AUDIT_MAX_BATCHES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "AUDIT_MAX_BATCHES must be a positive integer." >&2
  exit 2
fi

CONFIG="configs/samudra_multi_om4/train_halfdeg_coarse_latent_dynamics_proxy.yaml"
VALIDATION_CONFIG="configs/samudra_multi_om4/validate_cross_1_halfdeg_coarse_latent_dynamics.yaml"
arms=(
  "physical-only:1:0"
  "latent-only:0:1"
  "combined-001:1:0.01"
  "combined-01:1:0.1"
)

for specification in "${arms[@]}"; do
  IFS=: read -r arm _ _ <<<"${specification}"
  name="${RUN_PREFIX}-${arm}"
  for run_directory in \
    "${OUTPUT_BASE}/${name}" \
    "${OUTPUT_BASE}/${name}-cross-validation"; do
    if [[ -e "${run_directory}" ]]; then
      echo "Refusing to reuse existing run directory: ${run_directory}" >&2
      exit 5
    fi
  done
done

printf \
  'arm\ttrain_job_id\tvalidation_job_id\taudit_job_id\trun_name\tphysical_weight\tlatent_weight\n'
for specification in "${arms[@]}"; do
  IFS=: read -r arm physical_weight latent_weight <<<"${specification}"
  name="${RUN_PREFIX}-${arm}"
  run_directory="${OUTPUT_BASE}/${name}"
  validation_name="${name}-cross-validation"

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
  validation_args=(
    "--resume_ckpt_path=${run_directory}/saved_nets/best_validation_ckpt.pt"
    "--model.physical_forecast_loss_weight=${physical_weight}"
    "--model.latent_teacher_loss_weight=${latent_weight}"
    "--experiment.wandb.group=${WANDB_GROUP}-cross-validation"
    "--preemptible=true"
  )
  validation_args_string="${validation_args[*]}"
  validation_job_id="$(
    CONFIG="${VALIDATION_CONFIG}" \
    NAME="${validation_name}" \
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
    DATA_CACHE_DIR="${SCRATCH_DIR}/.data_cache/${validation_name}" \
      sbatch \
        --parsable \
        --dependency="afterok:${job_id}" \
        --chdir="${SCRATCH_DIR}" \
        --job-name="oe-s2-val-${arm}" \
        --account="torch_pr_347_courant" \
        --constraint="h200" \
        --gres="gpu:1" \
        --cpus-per-task="8" \
        --mem="64G" \
        --time="01:00:00" \
        --signal="B:USR1@120" \
        --requeue \
        --comment="preemption=yes;preemption_partitions_only=yes;requeue=true" \
        --output="${LOG_DIR}/${validation_name}-%j.out" \
        --error="${LOG_DIR}/${validation_name}-%j.err" \
        "${SBATCH_SCRIPT}"
  )"
  audit_job_id="$(
    RUN_DIR="${run_directory}" \
    CHECKPOINT="${run_directory}/saved_nets/best_validation_ckpt.pt" \
    INVERSE_CHECKPOINT="${INVERSE_CHECKPOINT}" \
    DATA_ROOT="${DATA_ROOT}" \
    SIF_PATH="${SIF_PATH}" \
    CODE_LAYER="${CODE_LAYER}" \
    AUDIT_SCRIPT="${AUDIT_SCRIPT}" \
    INVERSE_AUDIT_SCRIPT="${INVERSE_AUDIT_SCRIPT}" \
    MAX_BATCHES="${AUDIT_MAX_BATCHES}" \
      sbatch \
        --parsable \
        --dependency="afterok:${job_id}" \
        --chdir="${SCRATCH_DIR}" \
        --job-name="oe-s2-audit-${arm}" \
        --account="torch_pr_347_courant" \
        --constraint="h200" \
        --gres="gpu:1" \
        --cpus-per-task="8" \
        --mem="64G" \
        --time="00:30:00" \
        --requeue \
        --comment="preemption=yes;preemption_partitions_only=yes;requeue=true" \
        --output="${LOG_DIR}/${name}-audit-%j.out" \
        --error="${LOG_DIR}/${name}-audit-%j.err" \
        "${AUDIT_SBATCH_SCRIPT}"
  )"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${arm}" \
    "${job_id}" \
    "${validation_job_id}" \
    "${audit_job_id}" \
    "${name}" \
    "${physical_weight}" \
    "${latent_weight}"
done
