#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
: "${SCHEDULING_DIR:?}" "${STAGE:?}" "${WANDB_API_KEY:?}"
(cd "${SCHEDULING_DIR}" && sha256sum --check --status scheduling.sha256)
export BASE_NAME=2026-11-09-onedeg-tendency-scaled-2in1out
export BUNDLE_DIR="/scratch/am16581/runs/_experiments/${BASE_NAME}"
(cd "${BUNDLE_DIR}" && sha256sum --check --status bundle.sha256)
(cd "${BUNDLE_DIR}/stats" && sha256sum --check --status stats.sha256)
export CODE_COMMIT=acdbcf676867452e2f836dd7d4df3fa6477d76ee
export CODE_LAYER="/scratch/am16581/.apptainer-code-layers/samudra-code-${CODE_COMMIT}.img"
export SIF_PATH=/scratch/am16581/.apptainer-images/physicsnemo-26.05-4f86be45.sif
export IMAGE_REF=ghcr.io/m2lines/ocean-emulator-physicsnemo:26.05-4f86be45a09cb9eba1b51b6273e0189941b02495
export DATA_ROOT=/scratch/am16581/data/om4_onedeg_v3 OUTPUT_BASE=/scratch/am16581/runs
export SCRATCH_DIR=/scratch/am16581 SIF_DIR=/scratch/am16581/.apptainer-images
export APPTAINER_CACHEDIR=/scratch/am16581/apptainer-cache
export SINGULARITY_CACHEDIR=/scratch/am16581/singularity-cache
export NCCL_P2P_DISABLE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export REQUEUE_ON_USR1=1 WANDB_MODE=online PYTHONUNBUFFERED=1
CONFIG_ROOT=src/samudra/configs/experiments/onedeg_tendency_scaled
CONFIG_DIR="/opt/samudra-code/${CONFIG_ROOT}"

if [[ "${STAGE}" == 4gpu || "${STAGE}" == 8gpu ]]; then
  # Four ranks per node: one node starts the experiment; two finish it.
  export GPUS_PER_NODE=4
  export OMP_NUM_THREADS="$(( SLURM_CPUS_PER_TASK / GPUS_PER_NODE ))"
  export MKL_NUM_THREADS="${OMP_NUM_THREADS}" OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
  export NUMEXPR_NUM_THREADS="${OMP_NUM_THREADS}"
  export CONFIG="${CONFIG_ROOT}/train.yaml"
  export ARGS="--scheduler.target_epochs=70 --data.loading.num_workers=2"
  if [[ "${STAGE}" == 4gpu ]]; then
    export NAME="${BASE_NAME}-stage4gpu"
    export ARGS="${ARGS} --epochs=5 --gradient_accumulation_steps=2"
  else
    export NAME="${BASE_NAME}"
    export ARGS="${ARGS} --epochs=70 --gradient_accumulation_steps=1 --resume_ckpt_path=${OUTPUT_BASE}/${BASE_NAME}-stage4gpu/saved_nets/ckpt.pt"
  fi
  exec bash "${BUNDLE_DIR}/slurm_apptainer_train.sbatch"
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}" OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export APPTAINERENV_PYTHONPATH=/opt/samudra-code/src
export APPTAINERENV_PYTHONUNBUFFERED=1 APPTAINERENV_MPLBACKEND=Agg
export APPTAINERENV_WANDB_GIT_COMMIT="${CODE_COMMIT}"
export APPTAINERENV_SAMUDRA_CODE_COMMIT="${CODE_COMMIT}"
export APPTAINERENV_SAMUDRA_CODE_REPO_URL=https://github.com/m2lines/Samudra.git
export APPTAINERENV_SAMUDRA_CODE_LAYER_SHA256
APPTAINERENV_SAMUDRA_CODE_LAYER_SHA256="$(awk 'NR==1 {print $1}' "${CODE_LAYER}.sha256")"
export APPTAINERENV_SAMUDRA_CONTAINER_IMAGE_REF="${IMAGE_REF}"
export APPTAINERENV_SAMUDRA_CONTAINER_SIF_PATH="${SIF_PATH}"
container=(apptainer exec --overlay "${CODE_LAYER}:ro" --bind /scratch/am16581:/scratch/am16581 --pwd /opt/samudra-code "${SIF_PATH}" /workspace/.venv/bin/python)
case "${STAGE}" in
  interim-eval)
    export NAME="${BASE_NAME}-epoch5-eval"
    export CKPT_PATH="${OUTPUT_BASE}/${BASE_NAME}-stage4gpu/saved_nets/ema_ckpt.pt"
    srun --ntasks=1 apptainer exec --nv --overlay "${CODE_LAYER}:ro" \
      --bind /scratch/am16581:/scratch/am16581 --pwd /opt/samudra-code "${SIF_PATH}" \
      /workspace/.venv/bin/python "${BUNDLE_DIR}/one_step_skill.py" "${CONFIG_DIR}/eval.yaml" \
      "--experiment.name=${NAME}" "--ckpt_path=${CKPT_PATH}"
    export CONFIG="${CONFIG_ROOT}/eval.yaml"
    exec bash "${BUNDLE_DIR}/slurm_apptainer_eval.sbatch"
    ;;
  interim-analysis)
    srun --ntasks=1 "${container[@]}" -m samudra.metrics "${CONFIG_DIR}/metrics.yaml" \
      "--experiment.name=${BASE_NAME}-epoch5-eval"
    srun --ntasks=1 "${container[@]}" "${BUNDLE_DIR}/imprint_metrics.py" \
      --truth "${DATA_ROOT}/OM4.zarr" \
      --run "scaled-tendency-epoch5=${OUTPUT_BASE}/${BASE_NAME}-epoch5-eval/predictions.zarr" \
      --run "paper-samudra2=${OUTPUT_BASE}/2026-11-03-onedeg-original-eval/predictions.zarr" \
      --output "${OUTPUT_BASE}/${BASE_NAME}-epoch5-imprint"
    ;;
  interim-viz)
    runs="[{\"name\":\"scaled-tendency-epoch5\",\"location\":\"${OUTPUT_BASE}/${BASE_NAME}-epoch5-eval/predictions.zarr\"},{\"name\":\"paper-samudra2\",\"location\":\"${OUTPUT_BASE}/2026-11-03-onedeg-original-eval/predictions.zarr\"}]"
    srun --ntasks=1 "${container[@]}" -m samudra.viz "${CONFIG_DIR}/viz.yaml" \
      "--name=${BASE_NAME}-epoch5-viz" "--runs=${runs}" \
      '--steps=["timeseries_plots","short_timeseries_plots","global_thetao_time_series","global_salinity_timeseries_plots","ocean_temperature_profile_plots","ocean_salinity_profile_plots","thetao_mae_metrics"]'
    ;;
  *) echo "Unknown stage: ${STAGE}" >&2; exit 2 ;;
esac
