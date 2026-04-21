#!/bin/bash
# Slurm + Apptainer harness for distributed training, modeled after `skypilot/train.sky.yaml`.
#
# Submit with env vars (example):
#   export CONFIG=configs/.../train.yaml
#   export NAME_SUFFIX=om4_samudra_baseline   # NAME becomes YYYY-MM-DD-<suffix>
#   export ARGS="--epochs=2 --batch_size=1"
#   export WANDB_API_KEY=...   # optional; if unset, wandb is disabled
#   sbatch --nodes=1 --gres=gpu:rtx6000:8 scripts/slurm_apptainer_train.sbatch
#
# Multi-node example (requires enough nodes available):
#   sbatch --nodes=2 --gres=gpu:rtx6000:8 scripts/slurm_apptainer_train.sbatch
#
# Container selection (pick one):
#   export IMAGE_REF=ghcr.io/open-athena/ocean-emulator-physicsnemo:25.11-<hash>
#   export CONTAINER_HASH=<git sha>   # expands to 25.11-<sha>
#   export CONTAINER_TAG=25.11-latest
#
# Private GHCR pulls (if needed):
#   export GHCR_USERNAME=...
#   export GHCR_TOKEN=...
#
# Optional Nsight Systems profiling inside the container:
#   export NSYS_ARGS="--trace=cuda,nvtx,osrt --sample=none"
#   # If NSYS_ARGS omits -o/--output, reports are written under:
#   #   ${OUTPUT_BASE}/${NAME}/nsys/

#SBATCH --job-name=oe-train
#SBATCH --output=slurm-%j.out
#SBATCH --account=torch_pr_347_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# For 8-GPU runs on gr102, request full-node CPUs and QOS-capped memory.
#SBATCH --cpus-per-task=128
#SBATCH --mem=1400G
#SBATCH --gres=gpu:rtx6000:8
#SBATCH --time=8:00:00

set -euo pipefail

if command -v apptainer >/dev/null 2>&1; then
  APPTAINER_BIN=apptainer
elif command -v singularity >/dev/null 2>&1; then
  APPTAINER_BIN=singularity
else
  echo "Neither apptainer nor singularity is available on PATH." >&2
  exit 1
fi

CONFIG="${CONFIG:-}"
NAME="${NAME:-}"
NAME_SUFFIX="${NAME_SUFFIX:-}"
ARGS="${ARGS:-}"
NSYS_ARGS="${NSYS_ARGS:-}"
if [[ -z "${NAME}" && -n "${NAME_SUFFIX}" ]]; then
  NAME="$(date +%Y-%m-%d)-${NAME_SUFFIX}"
fi
if [[ -z "${CONFIG}" || -z "${NAME}" ]]; then
  echo "Missing required env vars. Need CONFIG and NAME." >&2
  echo "Tip: set NAME_SUFFIX to auto-generate NAME as YYYY-MM-DD-<suffix>." >&2
  echo "Example: export NAME_SUFFIX=om4_samudra_baseline" >&2
  exit 2
fi

# Config path inside the container image. We keep the env var as-is for
# backwards compatibility, but resolve relative paths under /workspace/
# (the image's working directory where configs/ is copied).
CONFIG_IN_CONTAINER="${CONFIG}"
if [[ "${CONFIG_IN_CONTAINER}" != /* ]]; then
  CONFIG_IN_CONTAINER="/workspace/${CONFIG_IN_CONTAINER}"
fi

# Data + outputs (host paths; will be bind-mounted into the container at same paths).
CURRENT_USER="${USER:-$(id -un)}"
DATA_ROOT_DEFAULT="/scratch/${CURRENT_USER}/data/om4_onedeg_v3"
OUTPUT_BASE_DEFAULT="/scratch/${CURRENT_USER}/runs"
DATA_ROOT="${DATA_ROOT:-${DATA_ROOT_DEFAULT}}"
OUTPUT_BASE="${OUTPUT_BASE:-${OUTPUT_BASE_DEFAULT}}"

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "DATA_ROOT directory does not exist: ${DATA_ROOT}" >&2
  echo "Set DATA_ROOT to an existing data directory." >&2
  echo "Example: export DATA_ROOT=${DATA_ROOT_DEFAULT}" >&2
  exit 5
fi
if [[ ! -d "${OUTPUT_BASE}" ]]; then
  echo "OUTPUT_BASE directory does not exist: ${OUTPUT_BASE}" >&2
  echo "Set OUTPUT_BASE to an existing output directory." >&2
  echo "Example: export OUTPUT_BASE=${OUTPUT_BASE_DEFAULT}" >&2
  exit 6
fi

RUN_DIR="${OUTPUT_BASE}/${NAME}"
if [[ -e "${RUN_DIR}" ]]; then
  echo "Refusing to run: output dir already exists: ${RUN_DIR}" >&2
  echo "Pick a unique NAME_SUFFIX." >&2
  exit 4
fi
NSYS_OUTPUT_DIR="${RUN_DIR}/nsys"

# W&B mode defaults: online if key provided, else disabled.
WANDB_MODE="${WANDB_MODE:-}"
if [[ -z "${WANDB_MODE}" ]]; then
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    WANDB_MODE="online"
  else
    WANDB_MODE="disabled"
  fi
fi

# Container selection (same conventions as scripts/slurm_apptainer_gpu_tests.sbatch).
CONTAINER_REGISTRY="${CONTAINER_REGISTRY:-ghcr.io}"
CONTAINER_ORG="${CONTAINER_ORG:-open-athena}"
CONTAINER_REPO="${CONTAINER_REPO:-ocean-emulator-physicsnemo}"
CONTAINER_HASH="${CONTAINER_HASH:-}"
CONTAINER_TAG="${CONTAINER_TAG:-}"
IMAGE_REF="${IMAGE_REF:-}"
if [[ -z "${IMAGE_REF}" ]]; then
  if [[ -z "${CONTAINER_TAG}" ]]; then
    if [[ -n "${CONTAINER_HASH}" ]]; then
      CONTAINER_TAG="25.11-${CONTAINER_HASH}"
    else
      CONTAINER_TAG="25.11-latest"
    fi
  fi
  IMAGE_REF="${CONTAINER_REGISTRY}/${CONTAINER_ORG}/${CONTAINER_REPO}:${CONTAINER_TAG}"
fi

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_DIR}"

SIF_DIR="${SIF_DIR:-${REPO_DIR}/.apptainer-images}"
mkdir -p "${SIF_DIR}"
# Default cache name keys off IMAGE_REF so distinct refs never collide.
IMAGE_REF_SAFE="$(echo "${IMAGE_REF}" | tr '/:@' '---')"
SIF_PATH="${SIF_PATH:-${SIF_DIR}/${IMAGE_REF_SAFE}.sif}"

# Apptainer tmp: keep on node-local /tmp to avoid scratch filesystems that don't support xattrs.
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/tmp}"

# GHCR auth (if image is private).
if [[ -n "${GHCR_USERNAME:-}" ]] && [[ -z "${APPTAINER_DOCKER_USERNAME:-}" ]]; then
  export APPTAINER_DOCKER_USERNAME="${GHCR_USERNAME}"
fi
if [[ -n "${GHCR_TOKEN:-}" ]] && [[ -z "${APPTAINER_DOCKER_PASSWORD:-}" ]]; then
  export APPTAINER_DOCKER_PASSWORD="${GHCR_TOKEN}"
fi

echo "Repo dir:   ${REPO_DIR}"
echo "Config:     ${CONFIG_IN_CONTAINER}"
echo "Name:       ${NAME}"
echo "Data root:  ${DATA_ROOT}"
echo "Outputs:    ${OUTPUT_BASE}"
echo "W&B mode:   ${WANDB_MODE}"
echo "Image ref:  ${IMAGE_REF}"
echo "SIF path:   ${SIF_PATH}"
if [[ -n "${NSYS_ARGS}" ]]; then
  echo "NSYS args:  ${NSYS_ARGS}"
  echo "NSYS out:   ${NSYS_OUTPUT_DIR}"
fi

if [[ ! -s "${SIF_PATH}" ]]; then
  echo "SIF not found locally. Pulling from registry..."
  "${APPTAINER_BIN}" pull "${SIF_PATH}" "docker://${IMAGE_REF}"
else
  echo "Using existing SIF."
fi

# Distributed rendezvous info (shared by all nodes).
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-8008}"
export MASTER_ADDR MASTER_PORT

# This harness launches one torchrun per node via srun. Each torchrun spawns one
# process per GPU on that node, and sets RANK/LOCAL_RANK/WORLD_SIZE via env://.
#
# Resource requests must match reality:
# - If you request 8 GPUs per node, torchrun will use 8 processes per node.
GPUS_PER_NODE="${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-}}"
if [[ -z "${GPUS_PER_NODE}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS_PER_NODE="$(nvidia-smi -L | wc -l | tr -d ' ')"
  else
    echo "Could not determine GPUS_PER_NODE (SLURM_GPUS_ON_NODE unset and nvidia-smi missing)." >&2
    exit 3
  fi
fi

# Avoid oversubscribing CPU threads when torchrun spawns one process per GPU.
if [[ -z "${OMP_NUM_THREADS:-}" ]]; then
  if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
    t="$(( SLURM_CPUS_PER_TASK / GPUS_PER_NODE ))"
    if (( t < 1 )); then
      t=1
    fi
    export OMP_NUM_THREADS="$t"
  else
    export OMP_NUM_THREADS=1
  fi
fi
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${OMP_NUM_THREADS}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${OMP_NUM_THREADS}}"
export PYTHONUNBUFFERED=1

# Create the run directory after preflight validation so early failures are
# retryable with the same NAME.
mkdir -p "${RUN_DIR}"

# ── Optional data staging to node-local NVMe ──
# Set STAGE_DATA=1 to copy zarr data from DATA_ROOT to fast node-local storage
# before training.  This avoids /scratch I/O stalls during training.
# STAGE_SOURCES: space-separated list of subdirectories to stage.
# STAGE_TIME_START / STAGE_TIME_END: optional time range to reduce copy size.
# STAGE_DST: destination directory on node-local storage.
STAGE_DATA="${STAGE_DATA:-0}"
STAGE_DST="${STAGE_DST:-/state/partition1/data}"
STAGE_SOURCES="${STAGE_SOURCES:-}"
STAGE_TIME_START="${STAGE_TIME_START:-}"
STAGE_TIME_END="${STAGE_TIME_END:-}"

if [[ "${STAGE_DATA}" == "1" && -n "${STAGE_SOURCES}" ]]; then
  echo ""
  echo "=== Staging data to node-local NVMe ==="
  echo "Source:      ${DATA_ROOT}"
  echo "Destination: ${STAGE_DST}"
  echo "Sources:     ${STAGE_SOURCES}"
  if [[ -n "${STAGE_TIME_START}" && -n "${STAGE_TIME_END}" ]]; then
    echo "Time range:  ${STAGE_TIME_START} to ${STAGE_TIME_END}"
  fi
  echo ""

  STAGE_BIND="${DATA_ROOT}:${DATA_ROOT},${STAGE_DST}:${STAGE_DST}"
  if [[ "${MOUNT_SOURCE:-1}" != "0" ]]; then
    STAGE_BIND="${STAGE_BIND},${REPO_DIR}/src:/workspace/src,${REPO_DIR}/scripts:/workspace/scripts"
  fi
  mkdir -p "${STAGE_DST}"

  STAGE_ARGS="--src-root ${DATA_ROOT} --dst-root ${STAGE_DST}"
  if [[ -n "${STAGE_TIME_START}" ]]; then
    STAGE_ARGS="${STAGE_ARGS} --time-start ${STAGE_TIME_START}"
  fi
  if [[ -n "${STAGE_TIME_END}" ]]; then
    STAGE_ARGS="${STAGE_ARGS} --time-end ${STAGE_TIME_END}"
  fi

  "${APPTAINER_BIN}" exec --nv \
    --bind "${STAGE_BIND}" \
    --pwd /workspace \
    "${SIF_PATH}" \
    bash -lc "
      export PYTHONPATH=/workspace/src:\${PYTHONPATH:-}
      /workspace/.venv/bin/python /workspace/scripts/stage_data.py \
        ${STAGE_ARGS} ${STAGE_SOURCES}
    "
  STAGE_EXIT=$?
  if [[ ${STAGE_EXIT} -ne 0 ]]; then
    echo "ERROR: Data staging failed (exit ${STAGE_EXIT}). Falling back to ${DATA_ROOT}." >&2
  else
    echo "Data staging complete. Switching DATA_ROOT to ${STAGE_DST}"
    DATA_ROOT="${STAGE_DST}"
  fi
fi

# Bind mounts — built after staging so DATA_ROOT reflects the (possibly local) path.
DATA_CACHE_DIR="${DATA_CACHE_DIR:-${REPO_DIR}/.data_cache/${NAME}}"
mkdir -p "${DATA_CACHE_DIR}"
binds=(
  "${DATA_ROOT}:${DATA_ROOT}"
  "${OUTPUT_BASE}:${OUTPUT_BASE}"
  "${DATA_CACHE_DIR}:/workspace/.data_cache"
)
if [[ "${MOUNT_SOURCE:-1}" != "0" ]]; then
  binds+=("${REPO_DIR}/src:/workspace/src")
  binds+=("${REPO_DIR}/configs:/workspace/configs")
fi
BIND_ARG="$(IFS=,; echo "${binds[*]}")"

# ── Optional periodic py-spy dump watchdog ──
# Set PYSPY_WATCHDOG=1 to enable periodic stack dumps of all training processes
# (including DataLoader workers).  Dumps land in ${RUN_DIR}/pyspy_dumps.log.
# PYSPY_INTERVAL controls the dump frequency in seconds (default: 60).
PYSPY_WATCHDOG="${PYSPY_WATCHDOG:-0}"
PYSPY_INTERVAL="${PYSPY_INTERVAL:-60}"
PYSPY_DUMP_LOG="${RUN_DIR}/pyspy_dumps.log"

# ── Optional py-spy record (flamegraph profiling) ──
# Set PYSPY_RECORD=1 to record a flamegraph/speedscope profile of the full run.
# Output lands in ${RUN_DIR}/pyspy_profile.<format>.
PYSPY_RECORD="${PYSPY_RECORD:-0}"
PYSPY_RECORD_RATE="${PYSPY_RECORD_RATE:-1}"
PYSPY_RECORD_FORMAT="${PYSPY_RECORD_FORMAT:-speedscope}"
PYSPY_RECORD_FILE="${RUN_DIR}/pyspy_profile.${PYSPY_RECORD_FORMAT}"

# ── Optional NVIDIA Nsight Systems (nsys) profiling ──
# Set NSYS_PROFILE=1 to wrap torchrun with nsys for GPU-level profiling.
# Use NSYS_DELAY to skip warmup (seconds) and NSYS_DURATION to limit capture.
# Mode: "lite" for low-overhead overview, "detailed" for full CUDA+Python backtraces.
NSYS_PROFILE="${NSYS_PROFILE:-0}"
NSYS_DELAY="${NSYS_DELAY:-120}"
NSYS_DURATION="${NSYS_DURATION:-120}"
NSYS_MODE="${NSYS_MODE:-lite}"
NSYS_OUTPUT="${RUN_DIR}/nsys_profile"

# ── Optional THP (Transparent Huge Pages) diagnostic monitoring ──
# Set THP_MONITOR=1 to periodically sample /proc/vmstat for THP and compaction
# counters. This diagnoses stalls caused by synchronous huge page defragmentation
# (see: /sys/kernel/mm/transparent_hugepage/defrag).
THP_MONITOR="${THP_MONITOR:-0}"
THP_MONITOR_INTERVAL="${THP_MONITOR_INTERVAL:-5}"
THP_VMSTAT_LOG="${RUN_DIR}/thp_vmstat.log"
THP_SETTINGS_LOG="${RUN_DIR}/thp_settings.log"

# ── Optional strace madvise tracing ──
# Set STRACE_MADVISE=1 to attach strace to one DataLoader worker process
# and capture madvise() syscalls (the trigger for THP defrag).
STRACE_MADVISE="${STRACE_MADVISE:-0}"
STRACE_LOG="${RUN_DIR}/strace_madvise.log"
if [[ -n "${NSYS_ARGS}" ]]; then
  mkdir -p "${NSYS_OUTPUT_DIR}"
fi

export OE_NSYS_ARGS="${NSYS_ARGS}"
export OE_NSYS_OUTPUT_DIR="${NSYS_OUTPUT_DIR}"

srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
  "${APPTAINER_BIN}" exec --nv \
    --bind "${BIND_ARG}" \
    --pwd /workspace \
    "${SIF_PATH}" \
    bash -lc "
      set -euo pipefail
      export PYTHONPATH=/workspace/src:\${PYTHONPATH:-}
      if [[ ! -x /workspace/.venv/bin/python ]]; then
        echo \"ERROR: /workspace/.venv/bin/python not found inside container\" >&2
        exit 1
      fi
      if [[ ! -f \"${CONFIG_IN_CONTAINER}\" ]]; then
        echo \"ERROR: config not found inside container: ${CONFIG_IN_CONTAINER}\" >&2
        echo \"Hint: rebuild/pull a container tag that includes /workspace/configs.\" >&2
        exit 2
      fi
      profiler_cmd=()
      if [[ -n \"\${OE_NSYS_ARGS:-}\" ]]; then
        if ! command -v nsys >/dev/null 2>&1; then
          echo \"ERROR: NSYS_ARGS was set, but nsys is not available inside the container PATH.\" >&2
          exit 3
        fi
        read -r -a nsys_args <<< \"\${OE_NSYS_ARGS}\"
        has_nsys_output=0
        for arg in \"\${nsys_args[@]}\"; do
          case \"\${arg}\" in
            -o|--output|-o?*|--output=*)
              has_nsys_output=1
              break
              ;;
          esac
        done
        profiler_cmd=(nsys profile \"\${nsys_args[@]}\")
        if [[ \"\${has_nsys_output}\" == \"0\" ]]; then
          profiler_cmd+=(
            -o \"\${OE_NSYS_OUTPUT_DIR}/nsys-node\${SLURM_NODEID:-0}-proc\${SLURM_PROCID:-0}\"
          )
        fi
        echo \"Nsight Systems profiling enabled on proc \${SLURM_PROCID:-unset}.\"
      fi
      TRAIN_CMD=\"\${profiler_cmd[*]} /workspace/.venv/bin/python -m torch.distributed.run \
        --nnodes=${SLURM_NNODES} \
        --nproc_per_node=${GPUS_PER_NODE} \
        --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} \
        --node_rank=\${SLURM_PROCID} \
        -m ocean_emulators.train ${CONFIG_IN_CONTAINER} \
        --experiment.name=${NAME} \
        --experiment.data_root=${DATA_ROOT} \
        --experiment.base_output_dir=${OUTPUT_BASE} \
        --experiment.wandb.mode=${WANDB_MODE} \
        --backend=nccl \
        ${ARGS}\"

      # Optionally wrap torchrun with nsys for GPU-level profiling.
      # nsys captures for NSYS_DURATION seconds after NSYS_DELAY, then stops
      # collection but lets the wrapped process continue running.
      if [[ \"${NSYS_PROFILE}\" == \"1\" ]]; then
        if [[ \"${NSYS_MODE}\" == \"detailed\" ]]; then
          NSYS_ARGS=\"--trace=cuda,nvtx,osrt,nccl --cudabacktrace=all --sample=none\"
        else
          NSYS_ARGS=\"--trace=cuda,nvtx,osrt,nccl --sample=none\"
        fi
        echo \"[nsys] Profiling: delay=${NSYS_DELAY}s duration=${NSYS_DURATION}s mode=${NSYS_MODE}\"
        echo \"[nsys] Output: ${NSYS_OUTPUT}\"
        nsys profile \
          \${NSYS_ARGS} \
          --delay=${NSYS_DELAY} \
          --duration=${NSYS_DURATION} \
          --output=\"${NSYS_OUTPUT}\" \
          --force-overwrite=true \
          \${TRAIN_CMD} &
        TORCHRUN_PID=\$!
        # nsys sends SIGTERM after duration expires. Trap it so training
        # continues and we just lose nsys (the profile is already saved).
      else
        \${TRAIN_CMD} &
        TORCHRUN_PID=\$!
      fi

      # ── THP settings snapshot ──
      if [[ \"${THP_MONITOR}\" == \"1\" ]]; then
        echo \"[thp] Logging THP kernel settings → ${THP_SETTINGS_LOG}\"
        {
          echo \"=== THP Settings at \$(date -Iseconds) ===\"
          echo \"--- /sys/kernel/mm/transparent_hugepage/enabled ---\"
          cat /sys/kernel/mm/transparent_hugepage/enabled 2>&1 || echo \"(not readable)\"
          echo \"--- /sys/kernel/mm/transparent_hugepage/defrag ---\"
          cat /sys/kernel/mm/transparent_hugepage/defrag 2>&1 || echo \"(not readable)\"
          echo \"--- /sys/kernel/mm/transparent_hugepage/khugepaged/defrag ---\"
          cat /sys/kernel/mm/transparent_hugepage/khugepaged/defrag 2>&1 || echo \"(not readable)\"
          echo \"--- /sys/kernel/mm/transparent_hugepage/khugepaged/scan_sleep_millisecs ---\"
          cat /sys/kernel/mm/transparent_hugepage/khugepaged/scan_sleep_millisecs 2>&1 || echo \"(not readable)\"
          echo \"--- Initial /proc/vmstat THP counters ---\"
          grep -E 'thp_|compact_|pgmigrate_' /proc/vmstat 2>&1 || echo \"(not readable)\"
          echo \"\"
        } > \"${THP_SETTINGS_LOG}\"

        echo \"[thp] Periodic vmstat sampling every ${THP_MONITOR_INTERVAL}s → ${THP_VMSTAT_LOG}\"
        (
          while kill -0 \${TORCHRUN_PID} 2>/dev/null; do
            {
              echo \"=== \$(date -Iseconds) ===\"
              grep -E 'thp_|compact_|pgmigrate_' /proc/vmstat 2>/dev/null || true
              # Also capture per-node memory info for NUMA awareness
              grep -E 'MemFree|MemAvailable|AnonHugePages|HugePages' /proc/meminfo 2>/dev/null || true
              echo \"\"
            } >> \"${THP_VMSTAT_LOG}\"
            sleep ${THP_MONITOR_INTERVAL}
          done
          echo \"[thp] torchrun exited, vmstat monitor stopping.\"
        ) &
        THP_MONITOR_PID=\$!
      fi

      # ── strace madvise on one DataLoader worker ──
      if [[ \"${STRACE_MADVISE}\" == \"1\" ]]; then
        echo \"[strace] Will attach to first DataLoader worker after 30s warmup → ${STRACE_LOG}\"
        (
          sleep 30  # Let DataLoader workers spawn
          # Find the first DataLoader worker child of torchrun
          DL_PID=\"\"
          for child in \$(pgrep -P \${TORCHRUN_PID} 2>/dev/null); do
            for grandchild in \$(pgrep -P \${child} 2>/dev/null); do
              # DataLoader workers are children of the training processes
              CMDLINE=\$(cat /proc/\${grandchild}/cmdline 2>/dev/null | tr '\\0' ' ')
              if echo \"\${CMDLINE}\" | grep -q 'dataloader\\|worker'; then
                DL_PID=\${grandchild}
                break 2
              fi
            done
          done

          # If we couldn't find by name, just pick the first grandchild
          if [[ -z \"\${DL_PID}\" ]]; then
            for child in \$(pgrep -P \${TORCHRUN_PID} 2>/dev/null | head -1); do
              DL_PID=\$(pgrep -P \${child} 2>/dev/null | head -1)
              break
            done
          fi

          if [[ -n \"\${DL_PID}\" ]]; then
            echo \"[strace] Attaching to PID \${DL_PID}\" >> \"${STRACE_LOG}\"
            # Trace madvise, mmap, mprotect — the syscalls involved in THP
            if command -v strace >/dev/null 2>&1; then
              strace -p \${DL_PID} -e trace=madvise,mmap,mprotect \
                -tt -T -f \
                -o \"${STRACE_LOG}\" 2>&1 &
              STRACE_PID=\$!
              echo \"[strace] Recording madvise/mmap/mprotect on PID \${DL_PID} (strace PID \${STRACE_PID})\"
            else
              echo \"[strace] strace not found in container, falling back to /proc/\${DL_PID}/syscall polling\"
              # Fallback: poll /proc/<pid>/syscall to detect if stuck in madvise
              while kill -0 \${DL_PID} 2>/dev/null && kill -0 \${TORCHRUN_PID} 2>/dev/null; do
                {
                  echo \"=== \$(date -Iseconds) PID=\${DL_PID} ===\"
                  cat /proc/\${DL_PID}/syscall 2>/dev/null || echo \"(unreadable)\"
                  cat /proc/\${DL_PID}/status 2>/dev/null | grep -E 'State|VmRSS|VmHWM|HugetlbPages' || true
                  # Check smaps for AnonHugePages
                  grep AnonHugePages /proc/\${DL_PID}/smaps_rollup 2>/dev/null || true
                } >> \"${STRACE_LOG}\"
                sleep 2
              done
            fi
          else
            echo \"[strace] Could not find a DataLoader worker PID\" >> \"${STRACE_LOG}\"
          fi
        ) &
        STRACE_BG_PID=\$!
      fi

      # ── py-spy dump watchdog ──
      if [[ \"${PYSPY_WATCHDOG}\" == \"1\" ]]; then
        echo \"[watchdog] Periodic py-spy dump every ${PYSPY_INTERVAL}s → ${PYSPY_DUMP_LOG}\"
        (
          PYSPY=/workspace/.venv/bin/py-spy
          if [[ ! -x \${PYSPY} ]]; then
            echo \"[watchdog] py-spy not found at \${PYSPY}, disabling watchdog\" >&2
            exit 0
          fi
          while kill -0 \${TORCHRUN_PID} 2>/dev/null; do
            {
              echo \"\"
              echo \"======================================\"
              echo \"=== \$(date -Iseconds) ===\"
              echo \"======================================\"
              timeout 30 \${PYSPY} dump --subprocesses --pid \${TORCHRUN_PID} 2>&1 || echo \"[watchdog] py-spy dump timed out or failed\"
            } >> \"${PYSPY_DUMP_LOG}\"
            sleep ${PYSPY_INTERVAL}
          done
          echo \"[watchdog] torchrun exited, watchdog stopping.\"
        ) &
        WATCHDOG_PID=\$!
      fi

      # ── py-spy record watchdog (windowed flamegraph snapshots) ──
      # Records 5-second windows at PYSPY_RECORD_RATE hz every PYSPY_INTERVAL seconds.
      # Each window produces a separate speedscope file for targeted analysis.
      if [[ \"${PYSPY_RECORD}\" == \"1\" ]]; then
        PYSPY_RECORD_DIR=\"${RUN_DIR}/pyspy_records\"
        mkdir -p \"\${PYSPY_RECORD_DIR}\" 2>/dev/null || true
        echo \"[record] py-spy windowed record: ${PYSPY_RECORD_RATE}hz × 5s every ${PYSPY_INTERVAL}s → \${PYSPY_RECORD_DIR}/\"
        (
          PYSPY=/workspace/.venv/bin/py-spy
          if [[ ! -x \${PYSPY} ]]; then
            echo \"[record] py-spy not found at \${PYSPY}, skipping\" >&2
            exit 0
          fi
          SEQ=0
          while kill -0 \${TORCHRUN_PID} 2>/dev/null; do
            STAMP=\$(date +%Y%m%dT%H%M%S)
            OUTFILE=\"\${PYSPY_RECORD_DIR}/\${STAMP}_\${SEQ}.${PYSPY_RECORD_FORMAT}\"
            \${PYSPY} record \
              --subprocesses \
              --pid \${TORCHRUN_PID} \
              --rate ${PYSPY_RECORD_RATE} \
              --format ${PYSPY_RECORD_FORMAT} \
              --duration 5 \
              --output \"\${OUTFILE}\" 2>&1 || echo \"[record] py-spy record failed for window \${SEQ}\"
            SEQ=\$((SEQ + 1))
            sleep ${PYSPY_INTERVAL}
          done
          echo \"[record] torchrun exited, record watchdog stopping.\"
        ) &
        RECORD_PID=\$!
      fi

      # Wait for torchrun to finish; propagate its exit code.
      wait \${TORCHRUN_PID}
      TORCHRUN_EXIT=\$?

      # Stop all background monitors.
      for BGPID in \"\${WATCHDOG_PID:-}\" \"\${RECORD_PID:-}\" \"\${THP_MONITOR_PID:-}\" \"\${STRACE_BG_PID:-}\" \"\${STRACE_PID:-}\"; do
        if [[ -n \"\${BGPID}\" ]]; then
          kill \${BGPID} 2>/dev/null || true
          wait \${BGPID} 2>/dev/null || true
        fi
      done

      # Final THP snapshot for comparison with initial.
      if [[ \"${THP_MONITOR}\" == \"1\" ]]; then
        {
          echo \"=== Final THP counters at \$(date -Iseconds) ===\"
          grep -E 'thp_|compact_|pgmigrate_' /proc/vmstat 2>/dev/null || true
          grep -E 'MemFree|MemAvailable|AnonHugePages|HugePages' /proc/meminfo 2>/dev/null || true
        } >> \"${THP_SETTINGS_LOG}\"
      fi

      exit \${TORCHRUN_EXIT}
    "
