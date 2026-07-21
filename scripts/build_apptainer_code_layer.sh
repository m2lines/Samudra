#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

# Build a small, read-only-at-runtime Apptainer overlay containing the Samudra
# source and configs from an exact, pushed git commit. The heavyweight runtime
# environment remains in a separately cached SIF.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  build_apptainer_code_layer.sh <40-character-git-commit> <runtime-sif>

Environment variables:
  CODE_REPO_URL       Git repository to fetch
                      (default: https://github.com/m2lines/Samudra.git)
  CODE_LAYER_DIR      Destination directory
                      (default: /scratch/$USER/.apptainer-code-layers)
  CODE_LAYER_SIZE_MB  EXT3 overlay capacity in MiB (default: 128)
  APPTAINER_BIN       apptainer or singularity executable (auto-detected)

The runtime SIF's /workspace/uv.lock and /workspace/pyproject.toml must be
byte-for-byte identical to the requested commit. A dependency change requires
a new runtime image; there is deliberately no mismatch override.
EOF
}

CODE_COMMIT="${1:-${CODE_COMMIT:-}}"
SIF_PATH="${2:-${SIF_PATH:-}}"
if [[ -z "${CODE_COMMIT}" || -z "${SIF_PATH}" ]]; then
  usage >&2
  exit 2
fi
if [[ ! "${CODE_COMMIT}" =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo "CODE_COMMIT must be a full 40-character git commit: ${CODE_COMMIT}" >&2
  exit 2
fi
CODE_COMMIT="${CODE_COMMIT,,}"

if [[ ! -s "${SIF_PATH}" ]]; then
  echo "Runtime SIF does not exist or is empty: ${SIF_PATH}" >&2
  exit 3
fi
SIF_PATH="$(realpath "${SIF_PATH}")"

if [[ -n "${APPTAINER_BIN:-}" ]]; then
  if ! command -v "${APPTAINER_BIN}" >/dev/null 2>&1; then
    echo "APPTAINER_BIN is not executable: ${APPTAINER_BIN}" >&2
    exit 4
  fi
elif command -v apptainer >/dev/null 2>&1; then
  APPTAINER_BIN=apptainer
elif command -v singularity >/dev/null 2>&1; then
  APPTAINER_BIN=singularity
else
  echo "Neither apptainer nor singularity is available on PATH." >&2
  exit 4
fi

for command_name in git python3 sha256sum flock; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Required command is unavailable: ${command_name}" >&2
    exit 4
  fi
done

CURRENT_USER="${USER:-$(id -un)}"
CODE_REPO_URL="${CODE_REPO_URL:-https://github.com/m2lines/Samudra.git}"
CODE_LAYER_DIR="${CODE_LAYER_DIR:-/scratch/${CURRENT_USER}/.apptainer-code-layers}"
CODE_LAYER_SIZE_MB="${CODE_LAYER_SIZE_MB:-128}"
if [[ ! "${CODE_LAYER_SIZE_MB}" =~ ^[1-9][0-9]*$ ]]; then
  echo "CODE_LAYER_SIZE_MB must be a positive integer: ${CODE_LAYER_SIZE_MB}" >&2
  exit 2
fi

mkdir -p "${CODE_LAYER_DIR}"
CODE_LAYER_DIR="$(realpath "${CODE_LAYER_DIR}")"
LAYER_BASENAME="samudra-code-${CODE_COMMIT}.img"
LAYER_PATH="${CODE_LAYER_DIR}/${LAYER_BASENAME}"
SHA_PATH="${LAYER_PATH}.sha256"
MANIFEST_PATH="${LAYER_PATH}.json"

# Only one process may publish a layer for a commit. The lock also prevents a
# reader from observing a partially copied sparse image.
exec 9>"${CODE_LAYER_DIR}/.${LAYER_BASENAME}.lock"
flock 9

BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/samudra-code-layer.XXXXXX")"
cleanup() {
  rm -rf "${BUILD_DIR}"
}
trap cleanup EXIT

CHECKOUT_DIR="${BUILD_DIR}/repo"
git init --quiet "${CHECKOUT_DIR}"
git -C "${CHECKOUT_DIR}" remote add origin "${CODE_REPO_URL}"
echo "Fetching pushed commit ${CODE_COMMIT} from ${CODE_REPO_URL}"
if ! git -C "${CHECKOUT_DIR}" fetch --quiet --no-tags --depth=1 origin "${CODE_COMMIT}"; then
  echo "Could not fetch ${CODE_COMMIT}; ensure the commit has been pushed and is reachable." >&2
  exit 5
fi
RESOLVED_COMMIT="$(git -C "${CHECKOUT_DIR}" rev-parse --verify FETCH_HEAD^{commit})"
if [[ "${RESOLVED_COMMIT}" != "${CODE_COMMIT}" ]]; then
  echo "Fetched commit ${RESOLVED_COMMIT}, expected ${CODE_COMMIT}." >&2
  exit 5
fi
git -C "${CHECKOUT_DIR}" checkout --quiet --detach "${RESOLVED_COMMIT}"

for source_path in src configs pyproject.toml uv.lock; do
  if [[ ! -e "${CHECKOUT_DIR}/${source_path}" ]]; then
    echo "Required source path is absent from ${CODE_COMMIT}: ${source_path}" >&2
    exit 6
  fi
done

SOURCE_MOUNT=/opt/samudra-layer-source
compare_runtime_file() {
  local relative_path="$1"
  if ! "${APPTAINER_BIN}" exec \
    --bind "${CHECKOUT_DIR}:${SOURCE_MOUNT}:ro" \
    "${SIF_PATH}" \
    cmp -s "${SOURCE_MOUNT}/${relative_path}" "/workspace/${relative_path}"; then
    echo "Runtime environment mismatch: ${relative_path} differs from the runtime SIF." >&2
    echo "Code commit: ${CODE_COMMIT}" >&2
    echo "Runtime SIF: ${SIF_PATH}" >&2
    echo "Rebuild the runtime image/SIF before using this commit." >&2
    echo "Source ${relative_path}:" >&2
    sha256sum "${CHECKOUT_DIR}/${relative_path}" >&2
    echo "Runtime ${relative_path}:" >&2
    "${APPTAINER_BIN}" exec "${SIF_PATH}" sha256sum "/workspace/${relative_path}" >&2 || true
    exit 7
  fi
}

compare_runtime_file uv.lock
compare_runtime_file pyproject.toml
echo "Runtime lockfiles match ${CODE_COMMIT}."

verify_existing_layer() {
  if [[ ! -s "${LAYER_PATH}" || ! -s "${SHA_PATH}" || ! -s "${MANIFEST_PATH}" ]]; then
    return 1
  fi
  (
    cd "${CODE_LAYER_DIR}"
    sha256sum --check --status "$(basename "${SHA_PATH}")"
  )
  local manifest_commit manifest_repo_url
  manifest_commit="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["code_commit"])' "${MANIFEST_PATH}")"
  manifest_repo_url="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["code_repo_url"])' "${MANIFEST_PATH}")"
  [[ "${manifest_commit}" == "${CODE_COMMIT}" && "${manifest_repo_url}" == "${CODE_REPO_URL}" ]]
}

if verify_existing_layer; then
  echo "Using existing verified code layer: ${LAYER_PATH}"
  echo "CODE_LAYER=${LAYER_PATH}"
  exit 0
fi
if [[ -e "${LAYER_PATH}" || -e "${SHA_PATH}" || -e "${MANIFEST_PATH}" ]]; then
  echo "Removing incomplete or invalid artifacts for ${CODE_COMMIT}." >&2
  rm -f "${LAYER_PATH}" "${SHA_PATH}" "${MANIFEST_PATH}"
fi

BUILD_TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
SOURCE_UV_LOCK_SHA256="$(sha256sum "${CHECKOUT_DIR}/uv.lock" | awk '{print $1}')"
SOURCE_PYPROJECT_SHA256="$(sha256sum "${CHECKOUT_DIR}/pyproject.toml" | awk '{print $1}')"
BUILD_MANIFEST="${BUILD_DIR}/source-manifest.json"
CODE_COMMIT="${CODE_COMMIT}" \
CODE_REPO_URL="${CODE_REPO_URL}" \
BUILD_TIMESTAMP="${BUILD_TIMESTAMP}" \
SOURCE_UV_LOCK_SHA256="${SOURCE_UV_LOCK_SHA256}" \
SOURCE_PYPROJECT_SHA256="${SOURCE_PYPROJECT_SHA256}" \
python3 - "${BUILD_MANIFEST}" <<'PY'
import json
import os
import sys

manifest = {
    "schema_version": 1,
    "code_commit": os.environ["CODE_COMMIT"],
    "code_repo_url": os.environ["CODE_REPO_URL"],
    "built_at_utc": os.environ["BUILD_TIMESTAMP"],
    "source_root": "/opt/samudra-code",
    "uv_lock_sha256": os.environ["SOURCE_UV_LOCK_SHA256"],
    "pyproject_sha256": os.environ["SOURCE_PYPROJECT_SHA256"],
    "included_paths": ["src", "configs", "pyproject.toml", "uv.lock"],
}
with open(sys.argv[1], "w", encoding="utf-8") as stream:
    json.dump(manifest, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

BUILD_LAYER="${BUILD_DIR}/${LAYER_BASENAME}"
echo "Creating ${CODE_LAYER_SIZE_MB} MiB sparse EXT3 code layer."
"${APPTAINER_BIN}" overlay create \
  --sparse \
  --size "${CODE_LAYER_SIZE_MB}" \
  --create-dir /opt/samudra-code \
  "${BUILD_LAYER}"

"${APPTAINER_BIN}" exec \
  --overlay "${BUILD_LAYER}" \
  --bind "${CHECKOUT_DIR}:${SOURCE_MOUNT}:ro" \
  --bind "${BUILD_MANIFEST}:/tmp/samudra-source-manifest.json:ro" \
  "${SIF_PATH}" \
  bash -lc '
    set -euo pipefail
    cp -a /opt/samudra-layer-source/src /opt/samudra-code/src
    cp -a /opt/samudra-layer-source/configs /opt/samudra-code/configs
    cp -a /opt/samudra-layer-source/pyproject.toml /opt/samudra-code/pyproject.toml
    cp -a /opt/samudra-layer-source/uv.lock /opt/samudra-code/uv.lock
    cp /tmp/samudra-source-manifest.json /opt/samudra-code/source-manifest.json
  '

# Verify the exact execution arrangement that the Slurm harness will use.
"${APPTAINER_BIN}" exec \
  --overlay "${BUILD_LAYER}:ro" \
  --pwd /opt/samudra-code \
  "${SIF_PATH}" \
  bash -lc '
    set -euo pipefail
    export PYTHONPATH=/opt/samudra-code/src
    test -d /opt/samudra-code/configs
    test -f /opt/samudra-code/src/samudra/__init__.py
    /workspace/.venv/bin/python -c \
      "import pathlib, samudra; assert pathlib.Path(samudra.__file__).is_relative_to(pathlib.Path(\"/opt/samudra-code\"))"
  '

LAYER_SHA256="$(sha256sum "${BUILD_LAYER}" | awk '{print $1}')"
STAGED_LAYER="${CODE_LAYER_DIR}/.${LAYER_BASENAME}.tmp.$$"
STAGED_SHA="${STAGED_LAYER}.sha256"
STAGED_MANIFEST="${STAGED_LAYER}.json"
cp --sparse=always "${BUILD_LAYER}" "${STAGED_LAYER}"
printf '%s  %s\n' "${LAYER_SHA256}" "${LAYER_BASENAME}" > "${STAGED_SHA}"
cp "${BUILD_MANIFEST}" "${STAGED_MANIFEST}"
chmod 0444 "${STAGED_LAYER}" "${STAGED_SHA}" "${STAGED_MANIFEST}"
mv "${STAGED_LAYER}" "${LAYER_PATH}"
mv "${STAGED_SHA}" "${SHA_PATH}"
mv "${STAGED_MANIFEST}" "${MANIFEST_PATH}"

echo "Published code layer: ${LAYER_PATH}"
echo "Layer SHA-256:       ${LAYER_SHA256}"
echo "CODE_LAYER=${LAYER_PATH}"
