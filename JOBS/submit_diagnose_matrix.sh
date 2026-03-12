#!/bin/bash
set -euo pipefail

# Submit a 2x2 diagnosis matrix for data pipeline bottlenecks.
# Matrix axes:
#   - DATA_NUM_WORKERS: 12, 24
#   - PIN_MEM: true, false
#
# Usage:
#   bash JOBS/submit_diagnose_matrix.sh
#   DRY_RUN=1 bash JOBS/submit_diagnose_matrix.sh
#
# Optional overrides:
#   GPUS=3 EPOCHS=1 SAVE_FREQ=1 GRAD_ACC=4 DDP_BUCKET_CAP_MB=25
#   DDP_USE_NO_SYNC=true DDP_BROADCAST_BUFFERS=false DDP_STATIC_GRAPH=false
#   DDP_FIND_UNUSED_PARAMETERS=false CONCURRENT_COMPUTE=false
#   WANDB_MODE=disabled EXTRA_ARGS="..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIAG_SCRIPT="${SCRIPT_DIR}/diagnose_train_llc.sh"

if [[ ! -f "${DIAG_SCRIPT}" ]]; then
  echo "Missing diagnose script: ${DIAG_SCRIPT}" >&2
  exit 1
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found on PATH." >&2
  exit 1
fi

# Shared defaults for all matrix runs (override via env)
GPUS="${GPUS:-3}"
EPOCHS="${EPOCHS:-1}"
SAVE_FREQ="${SAVE_FREQ:-1}"
GRAD_ACC="${GRAD_ACC:-4}"
DDP_BUCKET_CAP_MB="${DDP_BUCKET_CAP_MB:-25}"
DDP_USE_NO_SYNC="${DDP_USE_NO_SYNC:-true}"
DDP_BROADCAST_BUFFERS="${DDP_BROADCAST_BUFFERS:-false}"
DDP_STATIC_GRAPH="${DDP_STATIC_GRAPH:-false}"
DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-false}"
CONCURRENT_COMPUTE="${CONCURRENT_COMPUTE:-false}"
WANDB_MODE="${WANDB_MODE:-disabled}"
NCCL_DEBUG_LEVEL="${NCCL_DEBUG_LEVEL:-INFO}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
DRY_RUN="${DRY_RUN:-0}"

workers_values=(12 24)
pin_values=(true false)

stamp="$(date +%Y-%m-%d_%H%M%S)"
echo "Submitting diagnose matrix at ${stamp}"
echo "Shared settings: GPUS=${GPUS}, EPOCHS=${EPOCHS}, GRAD_ACC=${GRAD_ACC}, BUCKET=${DDP_BUCKET_CAP_MB}"

for workers in "${workers_values[@]}"; do
  for pin_mem in "${pin_values[@]}"; do
    short_pin="on"
    if [[ "${pin_mem}" == "false" ]]; then
      short_pin="off"
    fi

    job_name="diag_w${workers}_pin${short_pin}"
    exp_name="${stamp}-samudra_llc:diag:w${workers},pin_${short_pin},thermo_fields-Qnet-forcings,all_depths,extent=719"

    export_list="ALL,GPUS=${GPUS},EPOCHS=${EPOCHS},SAVE_FREQ=${SAVE_FREQ},GRAD_ACC=${GRAD_ACC},DDP_BUCKET_CAP_MB=${DDP_BUCKET_CAP_MB},DDP_USE_NO_SYNC=${DDP_USE_NO_SYNC},DDP_BROADCAST_BUFFERS=${DDP_BROADCAST_BUFFERS},DDP_STATIC_GRAPH=${DDP_STATIC_GRAPH},DDP_FIND_UNUSED_PARAMETERS=${DDP_FIND_UNUSED_PARAMETERS},DATA_NUM_WORKERS=${workers},PIN_MEM=${pin_mem},CONCURRENT_COMPUTE=${CONCURRENT_COMPUTE},WANDB_MODE=${WANDB_MODE},NCCL_DEBUG_LEVEL=${NCCL_DEBUG_LEVEL},EXPERIMENT_NAME=${exp_name}"

    if [[ -n "${EXTRA_ARGS}" ]]; then
      export_list+=" ,EXTRA_ARGS=${EXTRA_ARGS}"
      export_list="${export_list/ ,/,}"
    fi

    cmd=(sbatch --job-name "${job_name}" --export "${export_list}" "${DIAG_SCRIPT}")

    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "DRY_RUN: ${cmd[*]}"
    else
      out="$("${cmd[@]}")"
      echo "${job_name}: ${out}"
    fi
  done
done
