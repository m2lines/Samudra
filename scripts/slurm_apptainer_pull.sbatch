#!/bin/bash

# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

# CPU-only Slurm helper that pulls a container image into a reusable Apptainer
# SIF before GPU training/eval jobs need it.
#
# Container selection (pick one):
#   export IMAGE_REF=ghcr.io/m2lines/ocean-emulator-physicsnemo:26.05-<hash>
#   export CONTAINER_HASH=<git sha>   # expands to 26.05-<sha>
#   export CONTAINER_TAG=26.05-latest
#
# Optional output controls:
#   export SCRATCH_DIR=/scratch/$USER
#   export SIF_DIR=/scratch/$USER/.apptainer-images
#   export SIF_PATH=/scratch/$USER/.apptainer-images/physicsnemo.sif
#   export FORCE=1  # overwrite an existing SIF

#SBATCH --job-name=oe-pull-sif
#SBATCH --output=slurm-%j.out
#SBATCH --account=torch_pr_347_courant
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

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

for command_name in flock realpath; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Required command is unavailable: ${command_name}" >&2
    exit 1
  fi
done

CURRENT_USER="${USER:-$(id -un)}"
SCRATCH_DIR="${SCRATCH_DIR:-/scratch/${CURRENT_USER}}"
mkdir -p "${SCRATCH_DIR}"
SCRATCH_DIR="$(realpath "${SCRATCH_DIR}")"
cd "${SCRATCH_DIR}"

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

SIF_DIR="${SIF_DIR:-${SCRATCH_DIR}/.apptainer-images}"
mkdir -p "${SIF_DIR}"
SIF_DIR="$(realpath "${SIF_DIR}")"
IMAGE_REF_SAFE="$(echo "${IMAGE_REF}" | tr '/:@' '---')"
SIF_PATH="${SIF_PATH:-${SIF_DIR}/${IMAGE_REF_SAFE}.sif}"
SIF_PATH_DIR="$(dirname "${SIF_PATH}")"
mkdir -p "${SIF_PATH_DIR}"
SIF_PATH_DIR="$(realpath "${SIF_PATH_DIR}")"
SIF_PATH="${SIF_PATH_DIR}/$(basename "${SIF_PATH}")"

FORCE="${FORCE:-0}"
if [[ "${FORCE}" != "0" && "${FORCE}" != "1" ]]; then
  echo "FORCE must be 0 or 1: ${FORCE}" >&2
  exit 2
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
if [[ -n "${GHCR_USERNAME:-}" ]] && [[ -z "${SINGULARITY_DOCKER_USERNAME:-}" ]]; then
  export SINGULARITY_DOCKER_USERNAME="${GHCR_USERNAME}"
fi
if [[ -n "${GHCR_TOKEN:-}" ]] && [[ -z "${SINGULARITY_DOCKER_PASSWORD:-}" ]]; then
  export SINGULARITY_DOCKER_PASSWORD="${GHCR_TOKEN}"
fi

echo "Scratch:   ${SCRATCH_DIR}"
echo "Image ref: ${IMAGE_REF}"
echo "SIF path:  ${SIF_PATH}"

exec 9>"${SIF_PATH}.lock"
flock 9

if [[ -s "${SIF_PATH}" && "${FORCE}" == "0" ]]; then
  echo "Using existing SIF."
  echo "SIF_PATH=${SIF_PATH}"
  exit 0
fi

TMP_SIF="${SIF_PATH}.tmp.$$"
cleanup() {
  rm -f "${TMP_SIF}"
}
trap cleanup EXIT

rm -f "${TMP_SIF}"
echo "Pulling docker://${IMAGE_REF}"
"${APPTAINER_BIN}" pull "${TMP_SIF}" "docker://${IMAGE_REF}"
chmod 0444 "${TMP_SIF}"
mv -f "${TMP_SIF}" "${SIF_PATH}"
trap - EXIT

echo "Published SIF: ${SIF_PATH}"
echo "SIF_PATH=${SIF_PATH}"
