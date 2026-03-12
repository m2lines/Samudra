#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/orcd/home/002/codycruz/Ocean_Emulator}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${PROJECT_ROOT}/JOBS/train_llc.sh}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"

# Shared training knobs (forwarded to each spawned job)
DATA_NUM_WORKERS="${DATA_NUM_WORKERS:-10}"
PIN_MEM="${PIN_MEM:-false}"
DDP_BROADCAST_BUFFERS="${DDP_BROADCAST_BUFFERS:-false}"
DDP_TIMEOUT_MINUTES="${DDP_TIMEOUT_MINUTES:-30}"

# Per-job spatial knobs (edit defaults or override at submit time)
JOB1_FACE="${JOB1_FACE:-1}"
JOB1_I_START="${JOB1_I_START:-0}"
JOB1_I_END="${JOB1_I_END:-719}"
JOB1_J_START="${JOB1_J_START:-0}"
JOB1_J_END="${JOB1_J_END:-719}"

JOB2_FACE="${JOB2_FACE:-1}"
JOB2_I_START="${JOB2_I_START:-719}"
JOB2_I_END="${JOB2_I_END:-1439}"
JOB2_J_START="${JOB2_J_START:-0}"
JOB2_J_END="${JOB2_J_END:-719}"

JOB3_FACE="${JOB3_FACE:-1}"
JOB3_I_START="${JOB3_I_START:-0}"
JOB3_I_END="${JOB3_I_END:-719}"
JOB3_J_START="${JOB3_J_START:-719}"
JOB3_J_END="${JOB3_J_END:-1439}"

JOB4_FACE="${JOB4_FACE:-1}"
JOB4_I_START="${JOB4_I_START:-719}"
JOB4_I_END="${JOB4_I_END:-1439}"
JOB4_J_START="${JOB4_J_START:-719}"
JOB4_J_END="${JOB4_J_END:-1439}"

# Optional explicit names per job; defaults are auto-generated.
JOB1_NAME="${JOB1_NAME:-}"
JOB2_NAME="${JOB2_NAME:-}"
JOB3_NAME="${JOB3_NAME:-}"
JOB4_NAME="${JOB4_NAME:-}"

mkdir -p "${LOG_DIR}"

submit_one() {
  local idx="$1"
  local face="$2"
  local i_start="$3"
  local i_end="$4"
  local j_start="$5"
  local j_end="$6"
  local custom_name="$7"

  local default_name
  default_name="$(date +%Y-%m-%d)-samudra_llc_TEST:multi${idx}:face${face},i${i_start}-${i_end},j${j_start}-${j_end}"
  local job_name="${custom_name:-${default_name}}"
  local out_file="${LOG_DIR}/${job_name}-%j.out"

  local export_vars
  export_vars="ALL,DATA_NUM_WORKERS=${DATA_NUM_WORKERS},PIN_MEM=${PIN_MEM},DDP_BROADCAST_BUFFERS=${DDP_BROADCAST_BUFFERS},DDP_TIMEOUT_MINUTES=${DDP_TIMEOUT_MINUTES},LLC_FACE=${face},LLC_I_START=${i_start},LLC_I_END=${i_end},LLC_J_START=${j_start},LLC_J_END=${j_end}"

  local submit_out
  submit_out="$(
    sbatch \
      --cpus-per-task=15 \
      --mem=150GB \
      --gres=gpu:1 \
      --job-name="${job_name}" \
      --output="${out_file}" \
      --error="${out_file}" \
      --export="${export_vars}" \
      "${TRAIN_SCRIPT}"
  )"

  echo "${submit_out}"
}

echo "Submitting 4 single-GPU jobs from ${TRAIN_SCRIPT}"
echo "Shared knobs: workers=${DATA_NUM_WORKERS} pin_mem=${PIN_MEM} ddp_broadcast_buffers=${DDP_BROADCAST_BUFFERS} ddp_timeout_minutes=${DDP_TIMEOUT_MINUTES}"

submit_one 1 "${JOB1_FACE}" "${JOB1_I_START}" "${JOB1_I_END}" "${JOB1_J_START}" "${JOB1_J_END}" "${JOB1_NAME}"
submit_one 2 "${JOB2_FACE}" "${JOB2_I_START}" "${JOB2_I_END}" "${JOB2_J_START}" "${JOB2_J_END}" "${JOB2_NAME}"
submit_one 3 "${JOB3_FACE}" "${JOB3_I_START}" "${JOB3_I_END}" "${JOB3_J_START}" "${JOB3_J_END}" "${JOB3_NAME}"
submit_one 4 "${JOB4_FACE}" "${JOB4_I_START}" "${JOB4_I_END}" "${JOB4_J_START}" "${JOB4_J_END}" "${JOB4_NAME}"

