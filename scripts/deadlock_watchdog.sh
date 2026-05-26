#!/bin/bash

# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

# Deadlock watchdog: monitors a SLURM training job and dumps py-spy stack
# traces when training stops making progress.
#
# Watches the SLURM log file for new training steps.  When no new step appears
# for STALL_TIMEOUT seconds (default: 120), it finds all python worker PIDs
# inside the job's cgroup and runs `py-spy dump` on each one, saving the
# output to a timestamped file.
#
# Usage (run on the same node as the job, e.g. via srun or ssh):
#   bash scripts/deadlock_watchdog.sh <SLURM_JOB_ID>
#
# Optional env vars:
#   STALL_TIMEOUT=120    Seconds without progress before dumping (default: 120)
#   DUMP_DIR=<path>      Where to write dump files (default: next to slurm log)
#   MAX_DUMPS=3          Max number of dumps before giving up (default: 3)
#   DUMP_INTERVAL=60     Seconds between consecutive dumps (default: 60)
#   SIF_PATH=<path>      Path to .sif container image.  If set, py-spy is
#                         invoked via `apptainer exec` using the container's
#                         py-spy binary.  Apptainer shares the host PID
#                         namespace, so container py-spy can ptrace host PIDs.
#
# Prerequisites:
#   - py-spy must be available either on PATH or inside the container (set SIF_PATH)
#   - Must run as the same user or root (py-spy needs ptrace access)
#   - On compute nodes, you may need: echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope
#
# Example (from the submit node):
#   # Wait for job to start, then ssh to the compute node:
#   NODE=$(squeue -j 4784149 -o '%N' -h)
#
#   # If py-spy is on the host:
#   ssh $NODE "cd /scratch/$USER/Ocean_Emulator_multi && bash scripts/deadlock_watchdog.sh 4784149"
#
#   # If py-spy is only in the container:
#   SIF=$(ls /scratch/$USER/Ocean_Emulator_multi/.apptainer-images/*.sif | head -1)
#   ssh $NODE "cd /scratch/$USER/Ocean_Emulator_multi && SIF_PATH=$SIF bash scripts/deadlock_watchdog.sh 4784149"

set -euo pipefail

JOB_ID="${1:?Usage: $0 <SLURM_JOB_ID>}"

STALL_TIMEOUT="${STALL_TIMEOUT:-120}"
MAX_DUMPS="${MAX_DUMPS:-3}"
DUMP_INTERVAL="${DUMP_INTERVAL:-60}"
SIF_PATH="${SIF_PATH:-}"

# Find the SLURM log file.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
LOG_FILE="${SUBMIT_DIR}/slurm-${JOB_ID}.out"

if [[ ! -f "${LOG_FILE}" ]]; then
  echo "ERROR: Log file not found: ${LOG_FILE}" >&2
  echo "Make sure you're in the SLURM submit directory." >&2
  exit 1
fi

DUMP_DIR="${DUMP_DIR:-${SUBMIT_DIR}}"

# ── Resolve how to call py-spy ──
# Detect apptainer/singularity (needed for container-based py-spy).
APPTAINER_BIN=""
if command -v apptainer >/dev/null 2>&1; then
  APPTAINER_BIN=apptainer
elif command -v singularity >/dev/null 2>&1; then
  APPTAINER_BIN=singularity
fi

# Auto-detect SIF_PATH if not set: look in the .apptainer-images directory.
if [[ -z "${SIF_PATH}" ]]; then
  auto_sif="$(ls "${SUBMIT_DIR}"/.apptainer-images/*.sif 2>/dev/null | head -1)"
  if [[ -n "${auto_sif}" ]]; then
    SIF_PATH="${auto_sif}"
    echo "Auto-detected SIF: ${SIF_PATH}"
  fi
fi

if [[ -n "${SIF_PATH}" ]]; then
  # Use py-spy from inside the container.
  if [[ ! -f "${SIF_PATH}" ]]; then
    echo "ERROR: SIF file not found: ${SIF_PATH}" >&2
    exit 1
  fi
  if [[ -z "${APPTAINER_BIN}" ]]; then
    echo "ERROR: SIF_PATH is set but neither apptainer nor singularity is on PATH." >&2
    exit 1
  fi
  # Find py-spy inside the container.
  PYSPY_BIN=""
  if ${APPTAINER_BIN} exec "${SIF_PATH}" test -x /workspace/.venv/bin/py-spy 2>/dev/null; then
    PYSPY_BIN="/workspace/.venv/bin/py-spy"
  elif ${APPTAINER_BIN} exec "${SIF_PATH}" which py-spy >/dev/null 2>&1; then
    PYSPY_BIN="py-spy"
  else
    echo "ERROR: py-spy not found inside container at ${SIF_PATH}" >&2
    exit 1
  fi
  # Build the command: apptainer exec <SIF> <py-spy binary>
  # No extra flags like --pid or --bind needed — apptainer shares the host
  # PID namespace and mount namespace by default.
  PYSPY_CMD="${APPTAINER_BIN} exec ${SIF_PATH} ${PYSPY_BIN}"
  echo "Using py-spy from container: ${SIF_PATH} (${PYSPY_BIN})"
elif command -v py-spy >/dev/null 2>&1; then
  # Use host py-spy directly.
  PYSPY_CMD="py-spy"
  echo "Using py-spy from host PATH."
else
  echo "ERROR: py-spy not found on PATH and no .sif container found." >&2
  echo "Either install py-spy (pip install --user py-spy) or set SIF_PATH" >&2
  echo "to a container .sif that has py-spy installed." >&2
  exit 1
fi

echo ""
echo "=== Deadlock Watchdog ==="
echo "Job ID:        ${JOB_ID}"
echo "Log file:      ${LOG_FILE}"
echo "Stall timeout: ${STALL_TIMEOUT}s"
echo "Dump dir:      ${DUMP_DIR}"
echo "Max dumps:     ${MAX_DUMPS}"
echo "py-spy cmd:    ${PYSPY_CMD}"
echo ""

# Extract the latest training step number from the log.
get_latest_step() {
  grep -oP 'Training Epoch: \[\d+\]\s+\[\s*\K\d+' "${LOG_FILE}" | tail -1
}

# Find all python PIDs belonging to this SLURM job.
get_training_pids() {
  # First try: find via SLURM cgroup
  local cgroup_procs="/sys/fs/cgroup/cpu/slurm/uid_$(id -u)/job_${JOB_ID}/cgroup.procs"
  if [[ -f "${cgroup_procs}" ]]; then
    while read -r pid; do
      if [[ -f "/proc/${pid}/comm" ]] && grep -q python "/proc/${pid}/comm" 2>/dev/null; then
        echo "${pid}"
      fi
    done < "${cgroup_procs}"
    return
  fi

  # Fallback: find python processes whose command line references our training module.
  pgrep -u "$(id -u)" -f 'ocean_emulators.train' 2>/dev/null || true
}

# Dump stacks for all training PIDs.
do_dump() {
  local dump_num="$1"
  local timestamp
  timestamp="$(date +%Y%m%d_%H%M%S)"
  local dump_file="${DUMP_DIR}/pyspy_dump_${JOB_ID}_${timestamp}.txt"

  echo "[${timestamp}] Stall detected! Dumping stacks (dump ${dump_num}/${MAX_DUMPS})..."

  local pids
  pids="$(get_training_pids)"

  if [[ -z "${pids}" ]]; then
    echo "WARNING: No python training PIDs found for job ${JOB_ID}." >&2
    echo "The job may have already exited." >&2
    return 1
  fi

  {
    echo "=== py-spy deadlock dump ==="
    echo "Job ID:    ${JOB_ID}"
    echo "Timestamp: ${timestamp}"
    echo "Dump #:    ${dump_num}"
    echo "py-spy:    ${PYSPY_CMD}"
    echo ""

    while read -r pid; do
      echo "──────────────────────────────────────────────────────────"
      echo "PID: ${pid}  ($(cat /proc/${pid}/comm 2>/dev/null || echo '???'))"
      echo "cmdline: $(tr '\0' ' ' < /proc/${pid}/cmdline 2>/dev/null || echo '???')"
      echo "──────────────────────────────────────────────────────────"
      # --nonblocking: don't pause the target process.
      # --full-filenames: show complete file paths.
      ${PYSPY_CMD} dump --pid "${pid}" --nonblocking --full-filenames 2>&1 || \
        echo "(py-spy dump failed for PID ${pid})"
      echo ""
    done <<< "${pids}"
  } > "${dump_file}" 2>&1

  echo "Dump saved to: ${dump_file}"
  # Also print a summary to stdout so it shows in the watchdog's own log.
  echo "--- PIDs dumped ---"
  echo "${pids}" | while read -r pid; do echo "  ${pid}"; done
  echo "-------------------"
}

# ── Main loop ──
last_step="$(get_latest_step)"
last_progress_time="$(date +%s)"
dump_count=0

echo "Starting watchdog.  Current step: ${last_step:-<none>}"

while true; do
  # Check if the SLURM job is still running.
  if ! squeue -j "${JOB_ID}" -h -o '%T' 2>/dev/null | grep -qiE 'running|pending'; then
    echo "Job ${JOB_ID} is no longer running.  Exiting."
    break
  fi

  current_step="$(get_latest_step)"

  if [[ "${current_step}" != "${last_step}" ]]; then
    # Progress detected — reset the timer.
    last_step="${current_step}"
    last_progress_time="$(date +%s)"
  fi

  now="$(date +%s)"
  stall_duration=$(( now - last_progress_time ))

  if (( stall_duration >= STALL_TIMEOUT )); then
    dump_count=$(( dump_count + 1 ))

    if (( dump_count > MAX_DUMPS )); then
      echo "Reached max dumps (${MAX_DUMPS}).  Exiting watchdog."
      break
    fi

    do_dump "${dump_count}"

    # Reset timer so we wait DUMP_INTERVAL before the next dump.
    last_progress_time="$(date +%s)"
    STALL_TIMEOUT="${DUMP_INTERVAL}"
  fi

  sleep 10
done

echo "Watchdog finished.  Total dumps: ${dump_count}"
