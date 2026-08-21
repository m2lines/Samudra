#!/bin/bash

# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

# Slurm + Apptainer harness for distributed training, modeled after `skypilot/train.sky.yaml`.
#
# Submit with env vars (example):
#   export CONFIG=src/samudra/configs/.../train.yaml
#   export NAME_SUFFIX=om4_samudra_baseline   # NAME becomes YYYY-MM-DD-<suffix>
#   export ARGS="--epochs=2 --batch_size=1"
#   export WANDB_API_KEY=...   # optional; if unset, wandb is disabled
#   sbatch --nodes=1 --gres=gpu:rtx6000:8 scripts/slurm_apptainer_train.sbatch
#
# Multi-node example (requires enough nodes available):
#   sbatch --nodes=2 --gres=gpu:rtx6000:8 scripts/slurm_apptainer_train.sbatch
#
# Container selection (pick one):
#   export IMAGE_REF=ghcr.io/m2lines/ocean-emulator-physicsnemo:26.05-<hash>
#   export CONTAINER_HASH=<git sha>   # expands to 26.05-<sha>
#   export CONTAINER_TAG=26.05-latest
# Optional code layer built by scripts/build_apptainer_code_layer.sh:
#   export CODE_LAYER=/scratch/$USER/.apptainer-code-layers/samudra-code-<hash>.img
# Or a clean, commit-pinned checkout mounted read-only:
#   export CODE_DIR=/path/to/Samudra
#   export CODE_COMMIT=<full-git-sha>
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
#SBATCH --open-mode=append
#SBATCH --account=torch_pr_347_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# For 8-GPU runs on gr102, request full-node CPUs and QOS-capped memory.
#SBATCH --cpus-per-task=128
#SBATCH --mem=1400G
#SBATCH --gres=gpu:rtx6000:8
#SBATCH --time=8:00:00

set -euo pipefail

if ! command -v apptainer >/dev/null 2>&1 \
  && ! command -v singularity >/dev/null 2>&1; then
  if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck source=/etc/profile.d/modules.sh
    source /etc/profile.d/modules.sh
  fi
  if command -v module >/dev/null 2>&1; then
    module load "${APPTAINER_MODULE:-apptainer}"
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
NSYS_ARGS="${NSYS_ARGS:-}"
REQUEUE_ON_USR1="${REQUEUE_ON_USR1:-0}"
CODE_LAYER="${CODE_LAYER:-}"
CODE_DIR="${CODE_DIR:-}"
CODE_COMMIT="${CODE_COMMIT:-}"
if [[ -n "${CODE_LAYER}" && -n "${CODE_DIR}" ]]; then
  echo "Specify only one of CODE_LAYER or CODE_DIR." >&2
  exit 2
fi
if [[ -z "${NAME}" && -n "${NAME_SUFFIX}" ]]; then
  NAME="$(date +%Y-%m-%d)-${NAME_SUFFIX}"
fi
if [[ -z "${CONFIG}" || -z "${NAME}" ]]; then
  echo "Missing required env vars. Need CONFIG and NAME." >&2
  echo "Tip: set NAME_SUFFIX to auto-generate NAME as YYYY-MM-DD-<suffix>." >&2
  echo "Example: export NAME_SUFFIX=om4_samudra_baseline" >&2
  exit 2
fi

if [[ "${REQUEUE_ON_USR1}" == "1" ]]; then
  handle_requeue() {
    echo "Received USR1; requeueing Slurm job ${SLURM_JOB_ID}."
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
  }
  trap handle_requeue USR1
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
  if (( ${SLURM_RESTART_COUNT:-0} > 0 )); then
    echo "Requeued job detected; resuming from existing output dir: ${RUN_DIR}"
  else
    echo "Refusing to run: output dir already exists: ${RUN_DIR}" >&2
    echo "Pick a unique NAME_SUFFIX." >&2
    exit 4
  fi
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
CONTAINER_ORG="${CONTAINER_ORG:-m2lines}"
CONTAINER_REPO="${CONTAINER_REPO:-ocean-emulator-physicsnemo}"
CONTAINER_HASH="${CONTAINER_HASH:-}"
CONTAINER_TAG="${CONTAINER_TAG:-}"
IMAGE_REF="${IMAGE_REF:-}"
if [[ -z "${IMAGE_REF}" ]]; then
  if [[ -z "${CONTAINER_TAG}" ]]; then
    if [[ -n "${CONTAINER_HASH}" ]]; then
      CONTAINER_TAG="26.05-${CONTAINER_HASH}"
    else
      CONTAINER_TAG="26.05-latest"
    fi
  fi
  IMAGE_REF="${CONTAINER_REGISTRY}/${CONTAINER_ORG}/${CONTAINER_REPO}:${CONTAINER_TAG}"
fi

SCRATCH_DIR="${SCRATCH_DIR:-/scratch/${CURRENT_USER}}"
mkdir -p "${SCRATCH_DIR}"
SCRATCH_DIR="$(realpath "${SCRATCH_DIR}")"
cd "${SCRATCH_DIR}"

SIF_DIR="${SIF_DIR:-${SCRATCH_DIR}/.apptainer-images}"
mkdir -p "${SIF_DIR}"
# Default cache name keys off IMAGE_REF so distinct refs never collide.
IMAGE_REF_SAFE="$(echo "${IMAGE_REF}" | tr '/:@' '---')"
SIF_PATH_INPUT="${SIF_PATH:-}"
SIF_PATH="${SIF_PATH:-${SIF_DIR}/${IMAGE_REF_SAFE}.sif}"
CONTAINER_IMAGE_REF="${IMAGE_REF}"
if [[ -n "${SIF_PATH_INPUT}" && -s "${SIF_PATH_INPUT}" ]]; then
  # A pre-existing explicit SIF may not correspond to IMAGE_REF. Record its
  # path and avoid claiming the fallback pull reference as its identity.
  CONTAINER_IMAGE_REF=""
fi

# Apptainer tmp: keep on node-local /tmp to avoid scratch filesystems that don't support xattrs.
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/tmp}"

# GHCR auth (if image is private).
if [[ -n "${GHCR_USERNAME:-}" ]] && [[ -z "${APPTAINER_DOCKER_USERNAME:-}" ]]; then
  export APPTAINER_DOCKER_USERNAME="${GHCR_USERNAME}"
fi
if [[ -n "${GHCR_TOKEN:-}" ]] && [[ -z "${APPTAINER_DOCKER_PASSWORD:-}" ]]; then
  export APPTAINER_DOCKER_PASSWORD="${GHCR_TOKEN}"
fi

echo "Scratch:    ${SCRATCH_DIR}"
echo "Config:     ${CONFIG}"
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

# The SIF supplies the heavyweight Python/CUDA environment. Optional code can
# come from an EXT3 overlay or a clean, commit-pinned checkout mounted read-only.
CODE_ROOT=/workspace
CODE_GIT_COMMIT=""
CODE_GIT_REMOTE_URL=""
CODE_LAYER_SHA256=""
CODE_LAYER_MANIFEST=""
OVERLAY_ARGS=()
CODE_BIND_ARGS=()
if [[ -n "${CODE_DIR}" ]]; then
  if [[ -z "${CODE_COMMIT}" ]]; then
    echo "CODE_COMMIT is required when CODE_DIR is set." >&2
    exit 7
  fi
  if [[ ! "${CODE_COMMIT}" =~ ^[0-9a-fA-F]{40}$ ]]; then
    echo "CODE_COMMIT must be a full 40-character Git commit: ${CODE_COMMIT}" >&2
    exit 7
  fi
  if [[ ! -d "${CODE_DIR}" ]]; then
    echo "CODE_DIR does not exist or is not a directory: ${CODE_DIR}" >&2
    exit 7
  fi
  CODE_DIR="$(realpath "${CODE_DIR}")"
  if ! CODE_GIT_COMMIT="$(git -C "${CODE_DIR}" rev-parse --verify HEAD^{commit})"; then
    echo "CODE_DIR is not a Git checkout: ${CODE_DIR}" >&2
    exit 7
  fi
  CODE_GIT_COMMIT="${CODE_GIT_COMMIT,,}"
  if [[ "${CODE_GIT_COMMIT}" != "${CODE_COMMIT,,}" ]]; then
    echo "CODE_DIR HEAD does not match CODE_COMMIT." >&2
    echo "CODE_DIR HEAD: ${CODE_GIT_COMMIT}" >&2
    echo "CODE_COMMIT:    ${CODE_COMMIT,,}" >&2
    exit 7
  fi
  if [[ -n "$(git -C "${CODE_DIR}" status --porcelain)" ]]; then
    echo "CODE_DIR must be clean, including untracked files: ${CODE_DIR}" >&2
    exit 7
  fi
  for source_path in src pyproject.toml uv.lock; do
    if [[ ! -e "${CODE_DIR}/${source_path}" ]]; then
      echo "Required source path is absent from CODE_DIR: ${source_path}" >&2
      exit 7
    fi
  done
  CODE_GIT_REMOTE_URL="$(git -C "${CODE_DIR}" config --get remote.origin.url || true)"
  if [[ -z "${CODE_GIT_REMOTE_URL}" ]]; then
    echo "CODE_DIR does not have a remote.origin.url: ${CODE_DIR}" >&2
    exit 7
  fi
  CODE_ROOT=/opt/samudra-code
  CODE_BIND_ARGS=(--bind "${CODE_DIR}:${CODE_ROOT}:ro")
  export APPTAINERENV_SAMUDRA_CODE_DIR_PREFLIGHT=1
fi

# Read container identity and, for CODE_DIR, compare dependency files in one
# invocation. This avoids an additional expensive SIF expansion on systems
# without FUSE helpers.
if ! CONTAINER_METADATA_TEXT="$(
  "${APPTAINER_BIN}" exec \
    "${CODE_BIND_ARGS[@]}" \
    "${SIF_PATH}" \
    bash -lc '
      set -euo pipefail
      if [[ "${SAMUDRA_CODE_DIR_PREFLIGHT:-0}" == "1" ]]; then
        cmp -s /opt/samudra-code/uv.lock /workspace/uv.lock
        cmp -s /opt/samudra-code/pyproject.toml /workspace/pyproject.toml
      fi
      printf "%s\n%s\n" \
        "${SAMUDRA_CONTAINER_GIT_COMMIT:-${WANDB_GIT_COMMIT:-unknown}}" \
        "${SAMUDRA_CONTAINER_GIT_REMOTE_URL:-${WANDB_GIT_REMOTE_URL:-unknown}}"
    '
)"; then
  echo "Container metadata or CODE_DIR dependency compatibility check failed." >&2
  echo "CODE_DIR uv.lock and pyproject.toml must match the container SIF." >&2
  exit 7
fi
unset APPTAINERENV_SAMUDRA_CODE_DIR_PREFLIGHT
readarray -t CONTAINER_METADATA <<<"${CONTAINER_METADATA_TEXT}"
if [[ "${#CONTAINER_METADATA[@]}" -ne 2 ]]; then
  echo "Container has invalid commit/repository metadata: ${SIF_PATH}" >&2
  exit 7
fi
CONTAINER_GIT_COMMIT="${CONTAINER_METADATA[0]}"
CONTAINER_GIT_REMOTE_URL="${CONTAINER_METADATA[1]}"
if [[ -z "${CODE_DIR}" ]]; then
  CODE_GIT_COMMIT="${CONTAINER_GIT_COMMIT}"
  CODE_GIT_REMOTE_URL="${CONTAINER_GIT_REMOTE_URL}"
fi

if [[ -n "${CODE_LAYER}" ]]; then
  if [[ ! -s "${CODE_LAYER}" ]]; then
    echo "CODE_LAYER does not exist or is empty: ${CODE_LAYER}" >&2
    exit 7
  fi
  CODE_LAYER="$(realpath "${CODE_LAYER}")"
  CODE_LAYER_SHA_FILE="${CODE_LAYER}.sha256"
  CODE_LAYER_MANIFEST="${CODE_LAYER}.json"
  if [[ ! -s "${CODE_LAYER_SHA_FILE}" || ! -s "${CODE_LAYER_MANIFEST}" ]]; then
    echo "CODE_LAYER requires sidecars ${CODE_LAYER_SHA_FILE} and ${CODE_LAYER_MANIFEST}." >&2
    exit 7
  fi
  if ! (
    cd "$(dirname "${CODE_LAYER}")"
    sha256sum --check --status "$(basename "${CODE_LAYER_SHA_FILE}")"
  ); then
    echo "CODE_LAYER checksum verification failed: ${CODE_LAYER}" >&2
    exit 7
  fi
  if ! CODE_METADATA_TEXT="$(python3 -c '
import json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
print(manifest["code_commit"])
print(manifest["code_repo_url"])
' "${CODE_LAYER_MANIFEST}")"; then
    echo "Could not parse code-layer manifest: ${CODE_LAYER_MANIFEST}" >&2
    exit 7
  fi
  readarray -t CODE_METADATA <<<"${CODE_METADATA_TEXT}"
  if [[ "${#CODE_METADATA[@]}" -ne 2 ]]; then
    echo "Code-layer manifest has invalid commit/repository metadata: ${CODE_LAYER_MANIFEST}" >&2
    exit 7
  fi
  CODE_GIT_COMMIT="${CODE_METADATA[0]}"
  CODE_GIT_REMOTE_URL="${CODE_METADATA[1]}"
  CODE_LAYER_SHA256="$(awk 'NR == 1 {print $1}' "${CODE_LAYER_SHA_FILE}")"
  CODE_ROOT=/opt/samudra-code
  OVERLAY_ARGS=(--overlay "${CODE_LAYER}:ro")
  if ! "${APPTAINER_BIN}" exec \
    "${OVERLAY_ARGS[@]}" \
    --bind "${CODE_LAYER_MANIFEST}:/tmp/code-layer-manifest.json:ro" \
    "${SIF_PATH}" \
    bash -lc '
      cmp -s /tmp/code-layer-manifest.json /opt/samudra-code/source-manifest.json \
        && cmp -s /opt/samudra-code/uv.lock /workspace/uv.lock \
        && cmp -s /opt/samudra-code/pyproject.toml /workspace/pyproject.toml
    '; then
    echo "CODE_LAYER manifest or lockfiles do not match the selected container SIF." >&2
    echo "Code layer:  ${CODE_LAYER}" >&2
    echo "Container SIF: ${SIF_PATH}" >&2
    exit 7
  fi
fi

CONFIG_IN_CONTAINER="${CONFIG}"
if [[ "${CONFIG_IN_CONTAINER}" != /* ]]; then
  CONFIG_IN_CONTAINER="${CODE_ROOT}/${CONFIG_IN_CONTAINER}"
fi

echo "Resolved config: ${CONFIG_IN_CONTAINER}"
echo "Container commit: ${CONTAINER_GIT_COMMIT}"
echo "Code commit:    ${CODE_GIT_COMMIT}"
if [[ -n "${CODE_LAYER}" ]]; then
  echo "Code layer:     ${CODE_LAYER}"
  echo "Code layer SHA: ${CODE_LAYER_SHA256}"
elif [[ -n "${CODE_DIR}" ]]; then
  echo "Code directory: ${CODE_DIR} (read-only bind)"
else
  echo "Code source:    container SIF"
fi

# W&B's git panel must describe the code being executed, not the commit that
# happened to build the container image. Keep both identities explicitly.
export APPTAINERENV_WANDB_GIT_COMMIT="${CODE_GIT_COMMIT}"
export APPTAINERENV_WANDB_GIT_REMOTE_URL="${CODE_GIT_REMOTE_URL}"
export APPTAINERENV_SAMUDRA_CODE_COMMIT="${CODE_GIT_COMMIT}"
export APPTAINERENV_SAMUDRA_CODE_REPO_URL="${CODE_GIT_REMOTE_URL}"
export APPTAINERENV_SAMUDRA_CODE_LAYER_SHA256="${CODE_LAYER_SHA256}"
export APPTAINERENV_SAMUDRA_CODE_DIR="${CODE_DIR}"
export APPTAINERENV_SAMUDRA_CONTAINER_GIT_COMMIT="${CONTAINER_GIT_COMMIT}"
export APPTAINERENV_SAMUDRA_CONTAINER_GIT_REMOTE_URL="${CONTAINER_GIT_REMOTE_URL}"
export APPTAINERENV_SAMUDRA_CONTAINER_IMAGE_REF="${CONTAINER_IMAGE_REF}"
export APPTAINERENV_SAMUDRA_CONTAINER_SIF_PATH="${SIF_PATH}"
export SINGULARITYENV_WANDB_GIT_COMMIT="${CODE_GIT_COMMIT}"
export SINGULARITYENV_WANDB_GIT_REMOTE_URL="${CODE_GIT_REMOTE_URL}"
export SINGULARITYENV_SAMUDRA_CODE_COMMIT="${CODE_GIT_COMMIT}"
export SINGULARITYENV_SAMUDRA_CODE_REPO_URL="${CODE_GIT_REMOTE_URL}"
export SINGULARITYENV_SAMUDRA_CODE_LAYER_SHA256="${CODE_LAYER_SHA256}"
export SINGULARITYENV_SAMUDRA_CODE_DIR="${CODE_DIR}"
export SINGULARITYENV_SAMUDRA_CONTAINER_GIT_COMMIT="${CONTAINER_GIT_COMMIT}"
export SINGULARITYENV_SAMUDRA_CONTAINER_GIT_REMOTE_URL="${CONTAINER_GIT_REMOTE_URL}"
export SINGULARITYENV_SAMUDRA_CONTAINER_IMAGE_REF="${CONTAINER_IMAGE_REF}"
export SINGULARITYENV_SAMUDRA_CONTAINER_SIF_PATH="${SIF_PATH}"

# Distributed rendezvous info. Let torchrun atomically choose a free port for
# single-node jobs. Multi-node jobs need a shared, predetermined endpoint.
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-}"
TORCHRUN_STANDALONE=0
if [[ -z "${MASTER_PORT}" && "${SLURM_NNODES}" == "1" ]]; then
  TORCHRUN_STANDALONE=1
elif [[ -z "${MASTER_PORT}" ]]; then
  MASTER_PORT="$((15000 + SLURM_JOB_ID % 40000))"
fi
export MASTER_ADDR MASTER_PORT TORCHRUN_STANDALONE
if [[ "${TORCHRUN_STANDALONE}" == "1" ]]; then
  echo "Rendezvous: automatic free port (--standalone)"
else
  echo "Rendezvous: ${MASTER_ADDR}:${MASTER_PORT}"
fi

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
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export APPTAINERENV_PYTHONNOUSERSITE="${APPTAINERENV_PYTHONNOUSERSITE:-${PYTHONNOUSERSITE}}"
export SINGULARITYENV_PYTHONNOUSERSITE="${SINGULARITYENV_PYTHONNOUSERSITE:-${PYTHONNOUSERSITE}}"

# Bind mounts.
DATA_CACHE_DIR="${DATA_CACHE_DIR:-${SCRATCH_DIR}/.data_cache/${NAME}}"
mkdir -p "${DATA_CACHE_DIR}"
binds=(
  "${DATA_ROOT}:${DATA_ROOT}"
  "${OUTPUT_BASE}:${OUTPUT_BASE}"
  "${DATA_CACHE_DIR}:/workspace/.data_cache"
)
BIND_ARG="$(IFS=,; echo "${binds[*]}")"

# Create the run directory after preflight validation so early failures are
# retryable with the same NAME.
mkdir -p "${RUN_DIR}"
if [[ -n "${CODE_LAYER_MANIFEST}" ]]; then
  cp "${CODE_LAYER_MANIFEST}" "${RUN_DIR}/source-manifest.json"
elif [[ -n "${CODE_DIR}" ]]; then
  SAMUDRA_SOURCE_MANIFEST_PATH="${RUN_DIR}/source-manifest.json" \
  SAMUDRA_CODE_COMMIT="${CODE_GIT_COMMIT}" \
  SAMUDRA_CODE_REPO_URL="${CODE_GIT_REMOTE_URL}" \
  SAMUDRA_CODE_DIR="${CODE_DIR}" \
  SAMUDRA_UV_LOCK_SHA256="$(sha256sum "${CODE_DIR}/uv.lock" | awk '{print $1}')" \
  SAMUDRA_PYPROJECT_SHA256="$(sha256sum "${CODE_DIR}/pyproject.toml" | awk '{print $1}')" \
  python3 - <<'PY'
import datetime
import json
import os

manifest = {
    "schema_version": 1,
    "source_mode": "read_only_bind",
    "source_root": "/opt/samudra-code",
    "host_source_path": os.environ["SAMUDRA_CODE_DIR"],
    "code_commit": os.environ["SAMUDRA_CODE_COMMIT"],
    "code_repo_url": os.environ["SAMUDRA_CODE_REPO_URL"],
    "uv_lock_sha256": os.environ["SAMUDRA_UV_LOCK_SHA256"],
    "pyproject_sha256": os.environ["SAMUDRA_PYPROJECT_SHA256"],
    "included_paths": ["src", "pyproject.toml", "uv.lock"],
    "recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
with open(os.environ["SAMUDRA_SOURCE_MANIFEST_PATH"], "w", encoding="utf-8") as stream:
    json.dump(manifest, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY
fi
SAMUDRA_PROVENANCE_PATH="${RUN_DIR}/run-provenance.json" \
SAMUDRA_CODE_COMMIT="${CODE_GIT_COMMIT}" \
SAMUDRA_CODE_REPO_URL="${CODE_GIT_REMOTE_URL}" \
SAMUDRA_CODE_LAYER_SHA256="${CODE_LAYER_SHA256}" \
SAMUDRA_CODE_DIR="${CODE_DIR}" \
SAMUDRA_CONTAINER_GIT_COMMIT="${CONTAINER_GIT_COMMIT}" \
SAMUDRA_CONTAINER_GIT_REMOTE_URL="${CONTAINER_GIT_REMOTE_URL}" \
SAMUDRA_CONTAINER_IMAGE_REF="${CONTAINER_IMAGE_REF}" \
SAMUDRA_CONTAINER_SIF_PATH="${SIF_PATH}" \
python3 - <<'PY'
import json
import os

keys = (
    "SAMUDRA_CODE_COMMIT",
    "SAMUDRA_CODE_REPO_URL",
    "SAMUDRA_CODE_LAYER_SHA256",
    "SAMUDRA_CODE_DIR",
    "SAMUDRA_CONTAINER_GIT_COMMIT",
    "SAMUDRA_CONTAINER_GIT_REMOTE_URL",
    "SAMUDRA_CONTAINER_IMAGE_REF",
    "SAMUDRA_CONTAINER_SIF_PATH",
)
# Keep this compatible with the host Python 3.6 available on ORCD.
prefix = "SAMUDRA_"
provenance = {key[len(prefix) :].lower(): os.environ[key] for key in keys}
with open(os.environ["SAMUDRA_PROVENANCE_PATH"], "w", encoding="utf-8") as stream:
    json.dump(provenance, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY
if [[ -n "${NSYS_ARGS}" ]]; then
  mkdir -p "${NSYS_OUTPUT_DIR}"
fi

export OE_NSYS_ARGS="${NSYS_ARGS}"
export OE_NSYS_OUTPUT_DIR="${NSYS_OUTPUT_DIR}"

srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
  "${APPTAINER_BIN}" exec --nv \
    "${OVERLAY_ARGS[@]}" \
    "${CODE_BIND_ARGS[@]}" \
    --bind "${BIND_ARG}" \
    --pwd "${CODE_ROOT}" \
    "${SIF_PATH}" \
    bash -lc "
      set -euo pipefail
      export PYTHONNOUSERSITE=1
      export PYTHONPATH=${CODE_ROOT}/src:\${PYTHONPATH:-}
      if [[ ! -x /workspace/.venv/bin/python ]]; then
        echo \"ERROR: /workspace/.venv/bin/python not found inside container\" >&2
        exit 1
      fi
      if [[ ! -f \"${CONFIG_IN_CONTAINER}\" ]]; then
        echo \"ERROR: config not found inside container: ${CONFIG_IN_CONTAINER}\" >&2
        echo \"Hint: verify CONFIG and the selected code source.\" >&2
        exit 2
      fi
      torchrun_args=(
        --nnodes=${SLURM_NNODES}
        --nproc_per_node=${GPUS_PER_NODE}
        --node_rank=\${SLURM_PROCID}
      )
      if [[ \"\${TORCHRUN_STANDALONE}\" == \"1\" ]]; then
        torchrun_args+=(--standalone)
      else
        torchrun_args+=(
          --master_addr=\${MASTER_ADDR}
          --master_port=\${MASTER_PORT}
        )
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
      \"\${profiler_cmd[@]}\" /workspace/.venv/bin/python -m torch.distributed.run \
        \"\${torchrun_args[@]}\" \
        -m samudra.train \"${CONFIG_IN_CONTAINER}\" \
        --experiment.name=\"${NAME}\" \
        --experiment.data_root=\"${DATA_ROOT}\" \
        --experiment.base_output_dir=\"${OUTPUT_BASE}\" \
        --experiment.wandb.mode=\"${WANDB_MODE}\" \
        --backend=\"nccl\" \
        ${ARGS}
    "
