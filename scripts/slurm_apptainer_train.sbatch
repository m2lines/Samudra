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
#   export FORCE_PULL=1     # remove cached SIF for IMAGE_REF and repull
#
# Optional Nsight Systems profiling inside the container:
#   export NSYS_ARGS="--trace=cuda,nvtx,osrt --sample=none"
#   # If NSYS_ARGS omits -o/--output, reports are written under:
#   #   ${OUTPUT_BASE}/${NAME}/nsys/

#SBATCH --job-name=oe-train
#SBATCH --output=slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# For 8-GPU runs on gr102, request full-node CPUs and QOS-capped memory.
#SBATCH --cpus-per-task=128
#SBATCH --mem=1400G
#SBATCH --time=02:00:00

set -euo pipefail

if ! command -v apptainer >/dev/null 2>&1 \
  && ! command -v singularity >/dev/null 2>&1; then
  if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck source=/etc/profile.d/modules.sh
    source /etc/profile.d/modules.sh
  fi
  if command -v module >/dev/null 2>&1; then
    module load "${APPTAINER_MODULE:-apptainer/1.4.2}"
  fi
fi

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
ARGS_FILE="${ARGS_FILE:-}"
PRE_TRAIN_DIAGNOSTICS="${PRE_TRAIN_DIAGNOSTICS:-0}"
PRE_TRAIN_ZARR_PROBE_PATH="${PRE_TRAIN_ZARR_PROBE_PATH:-}"
PRE_TRAIN_ZARR_PROBE_VAR="${PRE_TRAIN_ZARR_PROBE_VAR:-}"
NSYS_ARGS="${NSYS_ARGS:-}"
FORCE_PULL="${FORCE_PULL:-0}"
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
echo "Force pull: ${FORCE_PULL}"
if [[ -n "${NSYS_ARGS}" ]]; then
  echo "NSYS args:  ${NSYS_ARGS}"
  echo "NSYS out:   ${NSYS_OUTPUT_DIR}"
fi

if [[ "${FORCE_PULL}" == "1" ]]; then
  echo "FORCE_PULL=1; removing cached SIF before pull."
  rm -f "${SIF_PATH}"
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

EXTRA_TRAIN_ARGS=()
if [[ -n "${ARGS}" ]]; then
  # Preserve literal quote characters in values like JSON overrides while still
  # splitting simple whitespace-delimited flags into separate argv entries.
  read -r -a EXTRA_TRAIN_ARGS <<< "${ARGS}"
fi
if [[ -n "${ARGS_FILE}" ]]; then
  if [[ ! -f "${ARGS_FILE}" ]]; then
    echo "ARGS_FILE does not exist: ${ARGS_FILE}" >&2
    exit 7
  fi
  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line}" ]] && continue
    EXTRA_TRAIN_ARGS+=("${line}")
  done < "${ARGS_FILE}"
fi

# Bind mounts.
DATA_CACHE_DIR="${DATA_CACHE_DIR:-${REPO_DIR}/.data_cache/${NAME}}"
mkdir -p "${DATA_CACHE_DIR}"
HOST_SSL_CERT_FILE="${HOST_SSL_CERT_FILE:-${SSL_CERT_FILE:-}}"
if [[ -z "${HOST_SSL_CERT_FILE}" ]]; then
  for candidate in \
    /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem \
    /etc/pki/tls/certs/ca-bundle.crt \
    /etc/ssl/certs/ca-certificates.crt \
    /etc/ssl/cert.pem; do
    if [[ -f "${candidate}" ]]; then
      HOST_SSL_CERT_FILE="${candidate}"
      break
    fi
  done
fi
if [[ -n "${HOST_SSL_CERT_FILE}" ]]; then
  resolved_host_ssl_cert_file="$(readlink -f "${HOST_SSL_CERT_FILE}" 2>/dev/null || true)"
  if [[ -n "${resolved_host_ssl_cert_file}" ]] && [[ -f "${resolved_host_ssl_cert_file}" ]]; then
    HOST_SSL_CERT_FILE="${resolved_host_ssl_cert_file}"
  fi
fi
if [[ -n "${HOST_SSL_CERT_FILE}" ]] && [[ ! -f "${HOST_SSL_CERT_FILE}" ]]; then
  echo "WARNING: host SSL_CERT_FILE does not exist: ${HOST_SSL_CERT_FILE}" >&2
  HOST_SSL_CERT_FILE=""
fi
CONTAINER_SSL_CERT_FILE=""
if [[ -n "${HOST_SSL_CERT_FILE}" ]]; then
  CONTAINER_SSL_CERT_FILE="/run/host-ca-certificates.pem"
  echo "Host CA bundle: ${HOST_SSL_CERT_FILE} -> ${CONTAINER_SSL_CERT_FILE}"
else
  echo "Host CA bundle: not found; falling back to container defaults/certifi"
fi
binds=(
  "${DATA_ROOT}:${DATA_ROOT}"
  "${OUTPUT_BASE}:${OUTPUT_BASE}"
  "${DATA_CACHE_DIR}:/workspace/.data_cache"
)
if [[ -n "${CONTAINER_SSL_CERT_FILE}" ]]; then
  binds+=("${HOST_SSL_CERT_FILE}:${CONTAINER_SSL_CERT_FILE}:ro")
fi
BIND_ARG="$(IFS=,; echo "${binds[*]}")"

# Create the run directory after preflight validation so early failures are
# retryable with the same NAME.
mkdir -p "${RUN_DIR}"
if [[ -n "${NSYS_ARGS}" ]]; then
  mkdir -p "${NSYS_OUTPUT_DIR}"
fi

export OE_CONFIG_IN_CONTAINER="${CONFIG_IN_CONTAINER}"
export OE_TRAIN_NAME="${NAME}"
export OE_DATA_ROOT="${DATA_ROOT}"
export OE_OUTPUT_BASE="${OUTPUT_BASE}"
export OE_WANDB_MODE="${WANDB_MODE}"
export OE_GPUS_PER_NODE="${GPUS_PER_NODE}"
export OE_PRE_TRAIN_DIAGNOSTICS="${PRE_TRAIN_DIAGNOSTICS}"
export OE_PRE_TRAIN_ZARR_PROBE_PATH="${PRE_TRAIN_ZARR_PROBE_PATH}"
export OE_PRE_TRAIN_ZARR_PROBE_VAR="${PRE_TRAIN_ZARR_PROBE_VAR}"
export OE_NSYS_ARGS="${NSYS_ARGS}"
export OE_NSYS_OUTPUT_DIR="${NSYS_OUTPUT_DIR}"
export OE_SSL_CERT_FILE_IN_CONTAINER="${CONTAINER_SSL_CERT_FILE}"

srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
  "${APPTAINER_BIN}" exec --nv \
    --bind "${BIND_ARG}" \
    --pwd /workspace \
    "${SIF_PATH}" \
    bash -lc '
      set -euo pipefail
      export PYTHONPATH=/workspace/src:${PYTHONPATH:-}
      if [[ ! -x /workspace/.venv/bin/python ]]; then
        echo "ERROR: /workspace/.venv/bin/python not found inside container" >&2
        exit 1
      fi
      configure_ssl_env() {
        local cert_file=""
        local candidate
        if [[ -n "${OE_SSL_CERT_FILE_IN_CONTAINER:-}" ]] \
          && [[ -f "${OE_SSL_CERT_FILE_IN_CONTAINER}" ]]; then
          cert_file="${OE_SSL_CERT_FILE_IN_CONTAINER}"
        else
          for candidate in \
            /etc/ssl/certs/ca-certificates.crt \
            /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem \
            /etc/pki/tls/certs/ca-bundle.crt \
            /etc/ssl/cert.pem; do
            if [[ -f "${candidate}" ]]; then
              cert_file="${candidate}"
              break
            fi
          done
        fi
        if [[ -z "${cert_file}" ]]; then
          cert_file="$(
            /workspace/.venv/bin/python -c "import certifi; print(certifi.where())" \
              2>/dev/null || true
          )"
        fi
        if [[ -n "${cert_file}" ]] && [[ -f "${cert_file}" ]]; then
          export SSL_CERT_FILE="${cert_file}"
          export REQUESTS_CA_BUNDLE="${cert_file}"
          export CURL_CA_BUNDLE="${cert_file}"
          if [[ -d /etc/ssl/certs ]]; then
            export SSL_CERT_DIR=/etc/ssl/certs
          elif [[ -d /etc/pki/tls/certs ]]; then
            export SSL_CERT_DIR=/etc/pki/tls/certs
          else
            unset SSL_CERT_DIR
          fi
          export OE_ACTIVE_SSL_CERT_FILE="${cert_file}"
        else
          unset SSL_CERT_FILE REQUESTS_CA_BUNDLE CURL_CA_BUNDLE SSL_CERT_DIR
          export OE_ACTIVE_SSL_CERT_FILE=""
          echo "WARNING: no CA bundle was found inside the container." >&2
        fi
      }
      configure_ssl_env
      if [[ ! -f "${OE_CONFIG_IN_CONTAINER}" ]]; then
        echo "ERROR: config not found inside container: ${OE_CONFIG_IN_CONTAINER}" >&2
        echo "Hint: rebuild/pull a container tag that includes /workspace/configs." >&2
        exit 2
      fi
      profiler_cmd=()
      if [[ -n "${OE_NSYS_ARGS}" ]]; then
        if ! command -v nsys >/dev/null 2>&1; then
          echo "ERROR: NSYS_ARGS was set, but nsys is not available inside the container PATH." >&2
          exit 3
        fi
        read -r -a nsys_args <<< "${OE_NSYS_ARGS}"
        has_nsys_output=0
        for arg in "${nsys_args[@]}"; do
          case "${arg}" in
            -o|--output|-o?*|--output=*)
              has_nsys_output=1
              break
              ;;
          esac
        done
        profiler_cmd=(nsys profile "${nsys_args[@]}")
        if [[ "${has_nsys_output}" == "0" ]]; then
          profiler_cmd+=(
            -o "${OE_NSYS_OUTPUT_DIR}/nsys-node${SLURM_NODEID:-0}-proc${SLURM_PROCID:-0}"
          )
        fi
        echo "Nsight Systems profiling enabled on proc ${SLURM_PROCID:-unset}."
      fi
      if [[ "${OE_PRE_TRAIN_DIAGNOSTICS}" == "1" && "${SLURM_PROCID}" == "0" ]]; then
        echo "===== Pre-train diagnostics start ====="
        echo "timestamp: $(date -Is)"
        echo "hostname: $(hostname)"
        echo "pwd: $(pwd)"
        echo "slurm_job_id: ${SLURM_JOB_ID:-unset}"
        echo "slurm_nodeid: ${SLURM_NODEID:-unset}"
        echo "slurm_procid: ${SLURM_PROCID:-unset}"
        echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-unset}"
        echo "data_root: ${OE_DATA_ROOT}"
        echo "ssl_cert_file: ${OE_ACTIVE_SSL_CERT_FILE:-unset}"
        echo "ssl_cert_dir: ${SSL_CERT_DIR:-unset}"
        echo "requests_ca_bundle: ${REQUESTS_CA_BUNDLE:-unset}"
        if [[ -n "${OE_ACTIVE_SSL_CERT_FILE:-}" ]]; then
          echo "+ ls -l ${OE_ACTIVE_SSL_CERT_FILE}"
          ls -l "${OE_ACTIVE_SSL_CERT_FILE}" || true
        fi
        if command -v nvidia-smi >/dev/null 2>&1; then
          echo "+ nvidia-smi -L"
          nvidia-smi -L || true
          echo "+ nvidia-smi --query-gpu=name,uuid,pci.bus_id,driver_version,memory.total --format=csv,noheader"
          nvidia-smi --query-gpu=name,uuid,pci.bus_id,driver_version,memory.total --format=csv,noheader || true
        fi
        if command -v findmnt >/dev/null 2>&1; then
          echo "+ findmnt -T ${OE_DATA_ROOT}"
          findmnt -T "${OE_DATA_ROOT}" -o TARGET,SOURCE,FSTYPE,OPTIONS || true
        fi
        if [[ -n "${OE_PRE_TRAIN_ZARR_PROBE_PATH}" ]] && command -v findmnt >/dev/null 2>&1; then
          echo "+ findmnt -T ${OE_PRE_TRAIN_ZARR_PROBE_PATH}"
          findmnt -T "${OE_PRE_TRAIN_ZARR_PROBE_PATH}" -o TARGET,SOURCE,FSTYPE,OPTIONS || true
        fi
        if [[ -f /etc/cufile.json ]]; then
          echo "+ cat /etc/cufile.json"
          cat /etc/cufile.json || true
        fi
        if command -v lsmod >/dev/null 2>&1; then
          echo "+ lsmod | grep -E \"nvidia_fs|nv_peer_mem|gds\""
          lsmod | grep -E "nvidia_fs|nv_peer_mem|gds" || true
        fi
        if command -v gdscheck >/dev/null 2>&1; then
          echo "+ gdscheck -p"
          gdscheck -p || true
        fi
        if command -v gds_stats >/dev/null 2>&1; then
          echo "+ gds_stats -h"
          gds_stats -h || true
        fi
        echo "+ python GDS probe"
        /workspace/.venv/bin/python - <<"PY" || true
import importlib
import os
from pathlib import Path

modules = ("cupy", "kvikio", "xarray", "zarr")
for module_name in modules:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "unknown")
        print(f"{module_name}.__version__={version}")
    except Exception as exc:
        print(f"{module_name}_import_error={type(exc).__name__}: {exc}")

try:
    from kvikio import defaults as kvikio_defaults

    compat_mode = kvikio_defaults.compat_mode
    compat_value = compat_mode() if callable(compat_mode) else compat_mode
    print(f"kvikio.compat_mode={compat_value}")
except Exception as exc:
    print(f"kvikio.compat_mode_error={type(exc).__name__}: {exc}")

try:
    import cupy

    device_count = cupy.cuda.runtime.getDeviceCount()
    print(f"cupy.device_count={device_count}")
    if device_count:
        props = cupy.cuda.runtime.getDeviceProperties(0)
        gpu_name = props.get("name", "unknown")
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode()
        print(f"cupy.device0.name={gpu_name}")
except Exception as exc:
    print(f"cupy_probe_error={type(exc).__name__}: {exc}")

probe_path = os.environ.get("OE_PRE_TRAIN_ZARR_PROBE_PATH", "")
probe_var = os.environ.get("OE_PRE_TRAIN_ZARR_PROBE_VAR", "")
if probe_path:
    probe_var_display = probe_var or "unset"
    print(f"probe.path={probe_path}")
    print(f"probe.var={probe_var_display}")
    try:
        import zarr

        from ocean_emulators.utils.location import _local_gds_store

        path = Path(probe_path)
        store = _local_gds_store(path)
        if store is None:
            print("probe.store_type=fallback_path")
            probe_store = str(path)
        else:
            print(
                "probe.store_type="
                f"{type(store).__module__}.{type(store).__qualname__}"
            )
            probe_store = store
        with zarr.config.enable_gpu():
            root = zarr.open_group(store=probe_store, mode="r")
            array_keys = list(root.array_keys())
            print(f"probe.array_count={len(array_keys)}")
            print(f"probe.array_keys_head={array_keys[:8]}")
            if probe_var:
                if probe_var in root:
                    array = root[probe_var]
                    array_chunks = getattr(array, "chunks", None)
                    print(f"probe.array_shape={array.shape}")
                    print(f"probe.array_chunks={array_chunks}")
                    index = tuple(slice(0, 1) for _ in array.shape)
                    sample = array[index]
                    print(
                        "probe.sample_type="
                        f"{type(sample).__module__}.{type(sample).__qualname__}"
                    )
                    device = getattr(sample, "device", None)
                    print(f"probe.sample_device={device}")
                else:
                    print("probe.array_missing=true")
    except Exception as exc:
        print(f"probe.error={type(exc).__name__}: {exc}")
PY
        echo "===== Pre-train diagnostics end ====="
      fi
      exec "${profiler_cmd[@]}" /workspace/.venv/bin/python -m torch.distributed.run \
        --nnodes='"${SLURM_NNODES}"' \
        --nproc_per_node="${OE_GPUS_PER_NODE}" \
        --master_addr='"${MASTER_ADDR}"' \
        --master_port='"${MASTER_PORT}"' \
        --node_rank="${SLURM_PROCID}" \
        -m ocean_emulators.train "${OE_CONFIG_IN_CONTAINER}" \
        --experiment.name="${OE_TRAIN_NAME}" \
        --experiment.data_root="${OE_DATA_ROOT}" \
        --experiment.base_output_dir="${OE_OUTPUT_BASE}" \
        --experiment.wandb.mode="${OE_WANDB_MODE}" \
        --backend="nccl" \
        --data.sources="[{\"data_location\":\"/orcd/home/002/jrusak/orcd/pool/data/LLC4320v3\",\"data_means_location\":\"/orcd/data/abodner/002/cody/LLC_means_stds/var_96_LLC_means.zarr\",\"data_stds_location\":\"/orcd/data/abodner/002/cody/LLC_means_stds/var_96_LLC_stds.zarr\"}]" \
        --experiment.prognostic_vars_key=single_1 \
        --experiment.boundary_vars_key=single_1 \
        "$@"
    ' bash "${EXTRA_TRAIN_ARGS[@]}"
