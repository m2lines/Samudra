#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
: "${BUNDLE_DIR:?}" "${SIF_PATH:?}" "${CODE_LAYER:?}" "${CODE_COMMIT:?}" "${PHASE:?}"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}" OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
(cd "${BUNDLE_DIR}" && sha256sum --check --status bundle.sha256)
CONFIG_DIR=/opt/samudra-code/src/samudra/configs/experiments/onedeg_tendency_scaled
if [[ "${PHASE}" == prepare ]]; then
  bash "${BUNDLE_DIR}/build_apptainer_code_layer.sh" "${CODE_COMMIT}" "${SIF_PATH}"
fi
(cd "$(dirname "${CODE_LAYER}")" && sha256sum --check --status "$(basename "${CODE_LAYER}.sha256")")
export APPTAINERENV_PYTHONPATH=/opt/samudra-code/src
export APPTAINERENV_PYTHONUNBUFFERED=1 APPTAINERENV_MPLBACKEND=Agg
container=(apptainer exec --overlay "${CODE_LAYER}:ro" --bind /scratch/am16581:/scratch/am16581 --pwd /opt/samudra-code "${SIF_PATH}" /workspace/.venv/bin/python)
case "${PHASE}" in
  prepare)
    original_stats=/scratch/am16581/runs/_experiments/2026-11-09-onedeg-tendency-residual-v2/stats
    (cd "${original_stats}" && sha256sum --check --status stats.sha256)
    test ! -e "${BUNDLE_DIR}/stats"
    mkdir "${BUNDLE_DIR}/stats"
    cp -a "${original_stats}/persistence_rms.zarr" "${original_stats}/stats.sha256" "${BUNDLE_DIR}/stats/"
    (cd "${BUNDLE_DIR}/stats" && sha256sum --check --status stats.sha256)
    "${container[@]}" "${BUNDLE_DIR}/preflight.py" "${CONFIG_DIR}/train.yaml" "${BUNDLE_DIR}"
    ;;
  train)
    (cd "${BUNDLE_DIR}/stats" && sha256sum --check --status stats.sha256)
    export CONFIG=src/samudra/configs/experiments/onedeg_tendency_scaled/train.yaml
    exec bash "${BUNDLE_DIR}/slurm_apptainer_train.sbatch"
    ;;
  eval)
    srun --ntasks=1 apptainer exec --nv --overlay "${CODE_LAYER}:ro" \
      --bind /scratch/am16581:/scratch/am16581 --pwd /opt/samudra-code "${SIF_PATH}" \
      /workspace/.venv/bin/python "${BUNDLE_DIR}/one_step_skill.py" "${CONFIG_DIR}/eval.yaml"
    export CONFIG=src/samudra/configs/experiments/onedeg_tendency_scaled/eval.yaml
    export TARGET_CHECKPOINT="${TRAIN_NAME}/saved_nets/ema_ckpt.pt"
    exec bash "${BUNDLE_DIR}/slurm_apptainer_eval.sbatch"
    ;;
  viz)
    srun --ntasks=1 "${container[@]}" -m samudra.viz "${CONFIG_DIR}/viz.yaml"
    ;;
  metrics)
    srun --ntasks=1 "${container[@]}" -m samudra.metrics "${CONFIG_DIR}/metrics.yaml"
    srun --ntasks=1 "${container[@]}" -m samudra.metrics "${CONFIG_DIR}/metrics.yaml" \
      --experiment.name=2026-11-03-onedeg-original-eval --model.pred_residuals=false \
      --model.residual_scale_location=null --data.output_steps=2
    ;;
  imprint)
    srun --ntasks=1 "${container[@]}" "${BUNDLE_DIR}/imprint_metrics.py" \
      --truth "${DATA_ROOT}/OM4.zarr" \
      --run "scaled-tendency-2in1out=${OUTPUT_BASE}/${TRAIN_NAME}-eval/predictions.zarr" \
      --run "paper-samudra2=${OUTPUT_BASE}/2026-11-03-onedeg-original-eval/predictions.zarr" \
      --output "${OUTPUT_BASE}/${TRAIN_NAME}-imprint"
    ;;
  *) echo "Unknown phase: ${PHASE}" >&2; exit 2 ;;
esac
