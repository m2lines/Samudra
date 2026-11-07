#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail
: "${ROOT:?}" "${CODE_LAYER:?}" "${SIF_PATH:?}"
export APPTAINER_TMPDIR="${SLURM_TMPDIR:-/tmp}"
export APPTAINER_CACHEDIR=/scratch/jr7309/apptainer-cache
export WANDB_CACHE_DIR="$ROOT/cache/wandb" MPLCONFIGDIR="$ROOT/cache/matplotlib"
cd "$(dirname "$CODE_LAYER")"
sha256sum -c "$(basename "$CODE_LAYER").sha256"
cd "$ROOT"
sha256sum -c inputs.sha256
run_python() {
    apptainer exec ${GPU_FLAG:-} --overlay "$CODE_LAYER:ro" --bind /scratch:/scratch "$SIF_PATH" \
        /usr/bin/env PYTHONPATH=/opt/samudra-code/src PYTHONNOUSERSITE=1 \
        PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 \
        /workspace/.venv/bin/python "$@"
}
case "$1" in
    qualify)
        run_python "$ROOT/qualify.py" "$ROOT"
        run_python -m pytest "$ROOT/test_loss.py" -q -m 'not cuda and not manual'
        ;;
    probe)
        export GPU_FLAG=--nv
        nvidia-smi
        for arm in legacy_mse wet_mse dynamic_wet; do
            run_python -m torch.distributed.run --standalone --nproc-per-node=4 \
                "$ROOT/probe.py" "$ROOT/configs/$arm.json"
        done
        touch "$ROOT/PROBE_PASSED"
        ;;
    *) exit 2 ;;
esac
