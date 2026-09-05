#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

# Build a SIF with OCI intermediates on allocation-local storage. This avoids
# consuming shared-scratch quota with a large, reproducible Apptainer cache.

set -euo pipefail

: "${IMAGE_REF:?IMAGE_REF is required}"
: "${SIF_PATH:?SIF_PATH is required}"

if ! command -v apptainer >/dev/null 2>&1 \
  && ! command -v singularity >/dev/null 2>&1; then
  module load "${APPTAINER_MODULE:-singularity-ce/4.3.3}"
fi
if command -v apptainer >/dev/null 2>&1; then
  apptainer_bin=apptainer
elif command -v singularity >/dev/null 2>&1; then
  apptainer_bin=singularity
else
  echo "Neither apptainer nor singularity is available" >&2
  exit 1
fi

node_root="${TMPDIR:-/tmp}/samudra-container-${SLURM_JOB_ID:-local}"
mkdir -p "$node_root" "$(dirname "$SIF_PATH")"
node_sif="$node_root/container.sif"
persistent_tmp="${SIF_PATH}.tmp.${SLURM_JOB_ID:-$$}"

cleanup() {
  rm -f "$persistent_tmp"
}
trap cleanup EXIT

exec 9>"${SIF_PATH}.lock"
flock 9
if [[ -s "$SIF_PATH" ]]; then
  echo "Using existing SIF: $SIF_PATH"
  exit 0
fi

export APPTAINER_TMPDIR="$node_root"
export SINGULARITY_TMPDIR="$node_root"
"$apptainer_bin" pull --disable-cache "$node_sif" "docker://$IMAGE_REF"
cp "$node_sif" "$persistent_tmp"
chmod 0444 "$persistent_tmp"
mv "$persistent_tmp" "$SIF_PATH"
trap - EXIT
echo "Published SIF: $SIF_PATH"
