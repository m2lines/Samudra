#!/bin/bash

# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

# Slurm + Apptainer harness for single-node evaluation, modeled after
# `skypilot/eval.sky.yaml` and `scripts/slurm_apptainer_train.sbatch`.
#
# Submit with env vars (example):
#   export CONFIG=configs/samudra_om4/eval.yaml
#   export NAME_SUFFIX=om4_samudra_baseline_eval  # NAME becomes YYYY-MM-DD-<suffix>
#   export TARGET_CHECKPOINT=2026-02-22-om4_samudra_baseline/saved_nets/ema_ckpt.pt
#   export ARGS="--num_model_steps_forward=25"
#   sbatch --gres=gpu:rtx6000:1 scripts/slurm_apptainer_eval.sbatch
#
# Container selection (pick one):
#   export IMAGE_REF=ghcr.io/m2lines/ocean-emulator-physicsnemo:26.05-<hash>
#   export CONTAINER_HASH=<git sha>   # expands to 26.05-<sha>
#   export CONTAINER_TAG=26.05-latest
# Optional code layer built by scripts/build_apptainer_code_layer.sh:
#   export CODE_LAYER=/scratch/$USER/.apptainer-code-layers/samudra-code-<hash>.img
#
# Checkpoint selection (pick one):
#   export TARGET_CHECKPOINT=<path-relative-to-OUTPUT_BASE>
#   export CKPT_PATH=/absolute/path/to/ema_ckpt.pt
#
# Private GHCR pulls (if needed):
#   export GHCR_USERNAME=...
#   export GHCR_TOKEN=...

#SBATCH --job-name=oe-eval
#SBATCH --output=slurm-%j.out
#SBATCH --account=torch_pr_347_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=04:00:00

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
BACKEND="${BACKEND:-cuda}"
CODE_LAYER="${CODE_LAYER:-}"
if [[ -z "${NAME}" && -n "${NAME_SUFFIX}" ]]; then
  NAME="$(date +%Y-%m-%d)-${NAME_SUFFIX}"
fi
if [[ -z "${CONFIG}" || -z "${NAME}" ]]; then
  echo "Missing required env vars. Need CONFIG and NAME." >&2
  echo "Tip: set NAME_SUFFIX to auto-generate NAME as YYYY-MM-DD-<suffix>." >&2
  echo "Example: export NAME_SUFFIX=om4_samudra_baseline_eval" >&2
  exit 2
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
  exit 3
fi
if [[ ! -d "${OUTPUT_BASE}" ]]; then
  echo "OUTPUT_BASE directory does not exist: ${OUTPUT_BASE}" >&2
  echo "Set OUTPUT_BASE to an existing output directory." >&2
  echo "Example: export OUTPUT_BASE=${OUTPUT_BASE_DEFAULT}" >&2
  exit 4
fi

RUN_DIR="${OUTPUT_BASE}/${NAME}"
if [[ -e "${RUN_DIR}" ]]; then
  echo "Refusing to run: output dir already exists: ${RUN_DIR}" >&2
  echo "Pick a unique NAME_SUFFIX." >&2
  exit 5
fi

# Resolve checkpoint path.
TARGET_CHECKPOINT="${TARGET_CHECKPOINT:-}"
CKPT_PATH="${CKPT_PATH:-}"
if [[ -n "${TARGET_CHECKPOINT}" && -n "${CKPT_PATH}" ]]; then
  echo "Specify only one of TARGET_CHECKPOINT or CKPT_PATH." >&2
  exit 6
fi
if [[ -n "${CKPT_PATH}" ]]; then
  if [[ "${CKPT_PATH}" = /* ]]; then
    CKPT_PATH_HOST="${CKPT_PATH}"
  else
    CKPT_PATH_HOST="$(realpath -m "${CKPT_PATH}")"
  fi
elif [[ -n "${TARGET_CHECKPOINT}" ]]; then
  if [[ "${TARGET_CHECKPOINT}" = /* ]]; then
    CKPT_PATH_HOST="${TARGET_CHECKPOINT}"
  else
    CKPT_PATH_HOST="${OUTPUT_BASE}/${TARGET_CHECKPOINT}"
  fi
else
  echo "Missing checkpoint. Set TARGET_CHECKPOINT (relative to OUTPUT_BASE) or CKPT_PATH." >&2
  exit 7
fi

if [[ ! -f "${CKPT_PATH_HOST}" ]]; then
  echo "Checkpoint not found: ${CKPT_PATH_HOST}" >&2
  exit 8
fi

# W&B mode defaults: online if key provided, else disabled.
WANDB_MODE="${WANDB_MODE:-}"
if [[ -z "${WANDB_MODE}" ]]; then
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    WANDB_MODE="online"
  else
    WANDB_MODE="disabled"
  fi
fi

# Container selection (same conventions as scripts/slurm_apptainer_train.sbatch).
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
echo "Checkpoint: ${CKPT_PATH_HOST}"
echo "W&B mode:   ${WANDB_MODE}"
echo "Backend:    ${BACKEND}"
echo "Image ref:  ${IMAGE_REF}"
echo "SIF path:   ${SIF_PATH}"

if [[ ! -s "${SIF_PATH}" ]]; then
  echo "SIF not found locally. Pulling from registry..."
  "${APPTAINER_BIN}" pull "${SIF_PATH}" "docker://${IMAGE_REF}"
else
  echo "Using existing SIF."
fi

CONTAINER_GIT_COMMIT="$(
  "${APPTAINER_BIN}" exec "${SIF_PATH}" \
    bash -lc 'printf %s "${SAMUDRA_CONTAINER_GIT_COMMIT:-${WANDB_GIT_COMMIT:-unknown}}"'
)"
CONTAINER_GIT_REMOTE_URL="$(
  "${APPTAINER_BIN}" exec "${SIF_PATH}" \
    bash -lc 'printf %s "${SAMUDRA_CONTAINER_GIT_REMOTE_URL:-${WANDB_GIT_REMOTE_URL:-unknown}}"'
)"
CODE_ROOT=/workspace
CODE_GIT_COMMIT="${CONTAINER_GIT_COMMIT}"
CODE_GIT_REMOTE_URL="${CONTAINER_GIT_REMOTE_URL}"
CODE_LAYER_SHA256=""
CODE_LAYER_MANIFEST=""
OVERLAY_ARGS=()
if [[ -n "${CODE_LAYER}" ]]; then
  if [[ ! -s "${CODE_LAYER}" ]]; then
    echo "CODE_LAYER does not exist or is empty: ${CODE_LAYER}" >&2
    exit 9
  fi
  CODE_LAYER="$(realpath "${CODE_LAYER}")"
  CODE_LAYER_SHA_FILE="${CODE_LAYER}.sha256"
  CODE_LAYER_MANIFEST="${CODE_LAYER}.json"
  if [[ ! -s "${CODE_LAYER_SHA_FILE}" || ! -s "${CODE_LAYER_MANIFEST}" ]]; then
    echo "CODE_LAYER requires sidecars ${CODE_LAYER_SHA_FILE} and ${CODE_LAYER_MANIFEST}." >&2
    exit 9
  fi
  if ! (
    cd "$(dirname "${CODE_LAYER}")"
    sha256sum --check --status "$(basename "${CODE_LAYER_SHA_FILE}")"
  ); then
    echo "CODE_LAYER checksum verification failed: ${CODE_LAYER}" >&2
    exit 9
  fi
  if ! CODE_METADATA_TEXT="$(python3 -c '
import json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
print(manifest["code_commit"])
print(manifest["code_repo_url"])
' "${CODE_LAYER_MANIFEST}")"; then
    echo "Could not parse code-layer manifest: ${CODE_LAYER_MANIFEST}" >&2
    exit 9
  fi
  readarray -t CODE_METADATA <<<"${CODE_METADATA_TEXT}"
  if [[ "${#CODE_METADATA[@]}" -ne 2 ]]; then
    echo "Code-layer manifest has invalid commit/repository metadata: ${CODE_LAYER_MANIFEST}" >&2
    exit 9
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
    exit 9
  fi
fi

CONFIG_IN_CONTAINER="${CONFIG}"
if [[ "${CONFIG_IN_CONTAINER}" != /* ]]; then
  CONFIG_IN_CONTAINER="${CODE_ROOT}/${CONFIG_IN_CONTAINER}"
fi

echo "Resolved config: ${CONFIG_IN_CONTAINER}"
echo "Container commit: ${CONTAINER_GIT_COMMIT}"
echo "Code commit:     ${CODE_GIT_COMMIT}"
if [[ -n "${CODE_LAYER}" ]]; then
  echo "Code layer:      ${CODE_LAYER}"
  echo "Code layer SHA:  ${CODE_LAYER_SHA256}"
else
  echo "Code layer:      none (using source baked into container SIF)"
fi

export APPTAINERENV_WANDB_GIT_COMMIT="${CODE_GIT_COMMIT}"
export APPTAINERENV_WANDB_GIT_REMOTE_URL="${CODE_GIT_REMOTE_URL}"
export APPTAINERENV_SAMUDRA_CODE_COMMIT="${CODE_GIT_COMMIT}"
export APPTAINERENV_SAMUDRA_CODE_REPO_URL="${CODE_GIT_REMOTE_URL}"
export APPTAINERENV_SAMUDRA_CODE_LAYER_SHA256="${CODE_LAYER_SHA256}"
export APPTAINERENV_SAMUDRA_CONTAINER_GIT_COMMIT="${CONTAINER_GIT_COMMIT}"
export APPTAINERENV_SAMUDRA_CONTAINER_GIT_REMOTE_URL="${CONTAINER_GIT_REMOTE_URL}"
export APPTAINERENV_SAMUDRA_CONTAINER_IMAGE_REF="${CONTAINER_IMAGE_REF}"
export APPTAINERENV_SAMUDRA_CONTAINER_SIF_PATH="${SIF_PATH}"
export SINGULARITYENV_WANDB_GIT_COMMIT="${CODE_GIT_COMMIT}"
export SINGULARITYENV_WANDB_GIT_REMOTE_URL="${CODE_GIT_REMOTE_URL}"
export SINGULARITYENV_SAMUDRA_CODE_COMMIT="${CODE_GIT_COMMIT}"
export SINGULARITYENV_SAMUDRA_CODE_REPO_URL="${CODE_GIT_REMOTE_URL}"
export SINGULARITYENV_SAMUDRA_CODE_LAYER_SHA256="${CODE_LAYER_SHA256}"
export SINGULARITYENV_SAMUDRA_CONTAINER_GIT_COMMIT="${CONTAINER_GIT_COMMIT}"
export SINGULARITYENV_SAMUDRA_CONTAINER_GIT_REMOTE_URL="${CONTAINER_GIT_REMOTE_URL}"
export SINGULARITYENV_SAMUDRA_CONTAINER_IMAGE_REF="${CONTAINER_IMAGE_REF}"
export SINGULARITYENV_SAMUDRA_CONTAINER_SIF_PATH="${SIF_PATH}"

if [[ -z "${OMP_NUM_THREADS:-}" ]]; then
  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
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

# If checkpoint lives outside DATA_ROOT/OUTPUT_BASE, bind its parent path too.
case "${CKPT_PATH_HOST}" in
  "${OUTPUT_BASE}"|"${OUTPUT_BASE}"/*|"${DATA_ROOT}"|"${DATA_ROOT}"/*)
    ;;
  *)
    CKPT_DIR="$(dirname "${CKPT_PATH_HOST}")"
    binds+=("${CKPT_DIR}:${CKPT_DIR}")
    ;;
esac

BIND_ARG="$(IFS=,; echo "${binds[*]}")"

# Create the run directory after preflight validation so early failures are
# retryable with the same NAME.
mkdir -p "${RUN_DIR}"
if [[ -n "${CODE_LAYER_MANIFEST}" ]]; then
  cp "${CODE_LAYER_MANIFEST}" "${RUN_DIR}/source-manifest.json"
fi
SAMUDRA_PROVENANCE_PATH="${RUN_DIR}/run-provenance.json" \
SAMUDRA_CODE_COMMIT="${CODE_GIT_COMMIT}" \
SAMUDRA_CODE_REPO_URL="${CODE_GIT_REMOTE_URL}" \
SAMUDRA_CODE_LAYER_SHA256="${CODE_LAYER_SHA256}" \
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
    "SAMUDRA_CONTAINER_GIT_COMMIT",
    "SAMUDRA_CONTAINER_GIT_REMOTE_URL",
    "SAMUDRA_CONTAINER_IMAGE_REF",
    "SAMUDRA_CONTAINER_SIF_PATH",
)
provenance = {key.removeprefix("SAMUDRA_").lower(): os.environ[key] for key in keys}
with open(os.environ["SAMUDRA_PROVENANCE_PATH"], "w", encoding="utf-8") as stream:
    json.dump(provenance, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

srun --ntasks=1 --ntasks-per-node=1 \
  "${APPTAINER_BIN}" exec --nv \
    "${OVERLAY_ARGS[@]}" \
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
        echo \"Hint: verify CONFIG and the selected code layer.\" >&2
        exit 2
      fi
      /workspace/.venv/bin/python -m samudra.eval \"${CONFIG_IN_CONTAINER}\" \
        --ckpt_path=\"${CKPT_PATH_HOST}\" \
        --experiment.name=\"${NAME}\" \
        --experiment.data_root=\"${DATA_ROOT}\" \
        --experiment.base_output_dir=\"${OUTPUT_BASE}\" \
        --experiment.wandb.mode=\"${WANDB_MODE}\" \
        --backend=\"${BACKEND}\" \
        ${ARGS}
    "
