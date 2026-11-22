#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=skyline-maps
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail
: "${SKYLINE_ROOT:?}" "${SKYLINE_COMMIT:?}" "${SKYLINE_ARCHIVE_SHA256:?}"
: "${SKYLINE_CONFIG:?}" "${SKYLINE_OUTPUT:?}" "${SIF_PATH:?}"
if [[ -f "${SKYLINE_OUTPUT}/COMPLETE.json" ]]; then exit 0; fi
requeue_job() { scontrol requeue "$SLURM_JOB_ID"; }
trap requeue_job USR1
export APPTAINER_CACHEDIR=/scratch/jr7309/apptainer-cache
export SINGULARITY_CACHEDIR=/scratch/jr7309/singularity-cache
export APPTAINER_TMPDIR="${SLURM_TMPDIR:-/tmp}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
mkdir -p "$SKYLINE_OUTPUT"
ARCHIVE="${SKYLINE_ROOT}/source-${SKYLINE_COMMIT}.tar"
actual=$(sha256sum "$ARCHIVE" | cut -d ' ' -f 1)
test "$actual" = "$SKYLINE_ARCHIVE_SHA256"
CODE_DIR=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/skyline-maps.XXXXXX")
trap 'rm -rf "$CODE_DIR"' EXIT
tar -xf "$ARCHIVE" -C "$CODE_DIR"
run_container() {
    apptainer exec --nv --bind "${CODE_DIR}:/opt/skyline-source:ro,/scratch:/scratch" \
      "$SIF_PATH" /usr/bin/env PYTHONPATH=/opt/skyline-source/src PYTHONNOUSERSITE=1 \
      TORCH_CUDNN_V8_API_DISABLED=1 OMP_NUM_THREADS=4 "$@" &
    child=$!
    while true; do
        status=0
        wait "$child" || status=$?
        if ! kill -0 "$child" 2>/dev/null; then return "$status"; fi
    done
}
# First pass caches the frozen global Samudra fields once for all map workers.
BASELINE_CACHE=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["baseline_cache"])' "$SKYLINE_CONFIG")
run_container /workspace/.venv/bin/python -m samudra.skyline.baseline \
    --config "$SKYLINE_CONFIG" --baseline-config "${SKYLINE_ROOT}/baseline-211241/train.yaml" \
    --checkpoint "${SKYLINE_ROOT}/baseline-211241/best_validation_ckpt.pt" --output "$BASELINE_CACHE"
nvidia-smi
run_container /workspace/.venv/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="${SKYLINE_WORLD_SIZE:-4}" \
    -m samudra.skyline.maps --config "$SKYLINE_CONFIG" --output "$SKYLINE_OUTPUT"
run_container /workspace/.venv/bin/python -m samudra.skyline.map_report "$SKYLINE_OUTPUT"
