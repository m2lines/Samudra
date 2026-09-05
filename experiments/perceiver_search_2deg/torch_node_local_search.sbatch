#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

# Stage the unchanged 2-degree store once per allocation, then run the search
# from the verified container and commit-specific source overlay.

set -euo pipefail

if ! command -v apptainer >/dev/null 2>&1 \
  && ! command -v singularity >/dev/null 2>&1; then
  if command -v module >/dev/null 2>&1; then
    module load "${APPTAINER_MODULE:-singularity-ce/4.3.3}"
  fi
fi
if command -v apptainer >/dev/null 2>&1; then
  apptainer_bin=apptainer
elif command -v singularity >/dev/null 2>&1; then
  apptainer_bin=singularity
else
  echo "Neither apptainer nor singularity is available" >&2
  exit 1
fi

: "${CODE_COMMIT:?CODE_COMMIT is required}"
: "${CODE_LAYER:?CODE_LAYER is required}"
: "${CONTAINER_SIF:?CONTAINER_SIF is required}"
: "${SOURCE_DATA_ROOT:?SOURCE_DATA_ROOT is required}"
: "${SEARCH_CONFIG:?SEARCH_CONFIG is required relative to /opt/samudra-code/src/samudra/configs}"

for path in "$CODE_LAYER" "$CONTAINER_SIF"; do
  if [[ ! -s "$path" ]]; then
    echo "Required container artifact is missing or empty: $path" >&2
    exit 2
  fi
done

layer_manifest="${CODE_LAYER}.json"
layer_commit="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["code_commit"])' "$layer_manifest")"
if [[ "$layer_commit" != "$CODE_COMMIT" ]]; then
  echo "Code layer commit $layer_commit does not match requested $CODE_COMMIT" >&2
  exit 3
fi

node_root="${TMPDIR:-/tmp}/samudra-search-${SLURM_JOB_ID:-local}"
staged_data_root="$node_root/data"
runtime_search="$node_root/search.yaml"
mkdir -p "$node_root"

if [[ ! -f "$staged_data_root/_SUCCESS" ]]; then
  test ! -e "$staged_data_root"
  mkdir -p "$staged_data_root"
  test -d "$SOURCE_DATA_ROOT/OM4.zarr"
  mkdir "$staged_data_root/OM4.zarr"
  find "$SOURCE_DATA_ROOT/OM4.zarr" -mindepth 1 -maxdepth 1 -print0 \
    | xargs -0 -n 1 -P "${STAGING_WORKERS:-16}" \
      cp -a -t "$staged_data_root/OM4.zarr" --
  for name in OM4_means.zarr OM4_stds.zarr; do
    test -d "$SOURCE_DATA_ROOT/$name"
    cp -a "$SOURCE_DATA_ROOT/$name" "$staged_data_root/$name"
  done
  touch "$staged_data_root/_SUCCESS"
fi

config_root=/opt/samudra-code/src/samudra/configs
sed \
  -e "s|__NODE_LOCAL_DATA_ROOT__|$staged_data_root|g" \
  -e "s|../search/osn-artifacts.yaml|$config_root/search/osn-artifacts.yaml|g" \
  "$config_root/$SEARCH_CONFIG" > "$runtime_search"

monitor_log="/home/${USER}/search-gpu-${SLURM_JOB_ID:-local}.dmon"
nvidia-smi dmon -s pucvmet -d 1 -o DT > "$monitor_log" &
monitor_pid=$!
cleanup() {
  kill "$monitor_pid" 2>/dev/null || true
}
trap cleanup EXIT

export WANDB_GIT_COMMIT="$CODE_COMMIT"
export PYTHONPATH=/opt/samudra-code/src

"$apptainer_bin" exec \
  --nv \
  --overlay "$CODE_LAYER:ro" \
  --bind "$node_root:$node_root" \
  --bind "/home/${USER}:/home/${USER}" \
  --pwd /opt/samudra-code \
  "$CONTAINER_SIF" \
  /workspace/.venv/bin/python -m samudra.search "$runtime_search"
