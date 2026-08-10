#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --job-name=2026-08-09-eval:Samudra_LLC:4-tile-blend=true,quintic,pre-blend_summary
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH -x node4100,node3401,node3000
#SBATCH -N 1
#SBATCH --mem=100GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH -G h200:1
#SBATCH --time=00-2:30:00
#SBATCH -o /orcd/home/002/codycruz/Ocean_Emulator/logs/%x-%j.out
#SBATCH -e /orcd/home/002/codycruz/Ocean_Emulator/logs/%x-%j.out
set -euo pipefail

module load miniforge/24.3.0-0

cd /orcd/home/002/codycruz/Ocean_Emulator
PYTHON_ENV_ROOT="${PYTHON_ENV_ROOT:-/orcd/home/002/codycruz/envs/ocean-emulators-py311-portable}"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON_ENV_ROOT}/bin/python}"
export PYTHONPATH="/orcd/home/002/codycruz/Ocean_Emulator/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: expected portable Python 3.11 environment at ${PYTHON_BIN}, but it is not executable." >&2
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ============== CHECKPOINT AND OUTPUT ==============
CKPT_PATH="${CKPT_PATH:-/orcd/data/abodner/002/cody/overflow/wandb_overflow/rb/2026-08-06:samudra_rb_llc:4-tile-base-experiment-19805916/saved_nets/ckpt_50.pt}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${SLURM_JOB_NAME:-$(basename "$0" .sh)}}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-/orcd/data/abodner/002/cody/inference_patch}"
EXPERIMENT_NAME="${EXPERIMENT_NAME}${SLURM_JOB_ID:+-${SLURM_JOB_ID}}"

# ============== DATA ==============
# DATA_LOCATION must be the DIRECTORY of tile caches, not a single .zarr.
# The tile catalog is built from each cache's absolute LLC x/y index arrays.
DATA_ROOT="${DATA_ROOT:-/orcd/data/abodner/002/cody/LLC_patch}"
DATA_LOCATION="${DATA_LOCATION:-720-div-4-test}"

INFER_START="${INFER_START:-2012-10-14}"
INFER_END="${INFER_END:-2012-10-17}"
INFERENCE_STRIDE="${INFERENCE_STRIDE:-1}"

# ============== MODEL ==============
# These must match how the checkpoint was trained. The live 4-tile run used the
# defaults below; group_norm_groups is resolved from the channel width, so it is
# left unset rather than forced.
MODEL_PAD="${MODEL_PAD:-constant}"
PRED_RESIDUALS="${PRED_RESIDUALS:-true}"

# ============== BLENDING ==============
# BLEND=false is the hard-crop control: tiles step independently and are simply
# cut and stitched. That is rung 2 of the ladder; BLEND=true is rung 3.
BLEND="${BLEND:-true}"
# WINDOW=quintic  -> smootherstep partition of unity (our operator, the default)
# WINDOW=kbd      -> Kaiser-Bessel-derived, STRATA's window
WINDOW="${WINDOW:-quintic}"
KBD_BETA="${KBD_BETA:-6.0}"
# RAMP_WIDTH unset -> taper over the overlap only (16 cells here).
# Set larger (e.g. 128) to window a whole tile the way STRATA does.
RAMP_WIDTH="${RAMP_WIDTH:-}"

# ============== DIAGNOSTIC: PRE-BLEND DISAGREEMENT (default off) ==============
# none    -> write nothing extra
# summary -> preblend.zarr with RMS |delta_A - delta_B| per channel per offset
#            across each seam. Small; this is what notebook 2 reads.
# full    -> additionally store the raw overlap-band residual differences. Large.
PREBLEND_MODE="${PREBLEND_MODE:-summary}"

# ============== REPACK TO 4D ==============
# The rollout writes flat channels (Theta_0 ... Theta_50) because that is the
# model's own layout. Repacking to Theta(time, k, lat, lon) is what makes the
# output loadable by the emulator-comparison notebooks. Roughly doubles the
# footprint, so set REPACK=false to skip, or narrow REPACK_FIELDS.
# Note: Eta has no depth index and so does not survive the repack; read it from
# the flat predictions.zarr, which is kept either way.
REPACK="${REPACK:-true}"
REPACK_FIELDS="${REPACK_FIELDS:-U V Theta Salt}"
REPACK_OVERWRITE="${REPACK_OVERWRITE:-false}"

echo "======== tiled (2x2) blended rollout ========"
echo "checkpoint:         ${CKPT_PATH}"
echo "tile cache dir:     ${DATA_ROOT}/${DATA_LOCATION}"
echo "inference window:   ${INFER_START} -> ${INFER_END} (stride ${INFERENCE_STRIDE})"
echo "blend:              ${BLEND} (window=${WINDOW}, ramp_width=${RAMP_WIDTH:-overlap})"
echo "preblend_mode:      ${PREBLEND_MODE}"
echo "perturbation:       ${PERTURBATION}"
echo "output:             ${BASE_OUTPUT_DIR}/${EXPERIMENT_NAME}"
echo

TILING_ARGS=(
  --tiling.blend "${BLEND}"
  --tiling.window "${WINDOW}"
  --tiling.kbd_beta "${KBD_BETA}"
  --tiling.preblend_mode "${PREBLEND_MODE}"
  --tiling.perturbation "${PERTURBATION}"
  --tiling.perturbation_amplitude "${PERTURBATION_AMPLITUDE}"
  --tiling.perturbation_box "${PERTURBATION_BOX}"
  --tiling.perturbation_channel "${PERTURBATION_CHANNEL}"
  --tiling.response_bins "${RESPONSE_BINS}"
)
if [[ -n "${RAMP_WIDTH}" ]]; then
  TILING_ARGS+=(--tiling.ramp_width "${RAMP_WIDTH}")
fi

MODEL_ARGS=()
if [[ -n "${MODEL_PAD}" ]]; then
  MODEL_ARGS+=(--model.pad "${MODEL_PAD}")
fi
if [[ -n "${PRED_RESIDUALS}" ]]; then
  MODEL_ARGS+=(--model.pred_residuals "${PRED_RESIDUALS}")
fi

"${PYTHON_BIN}" -m ocean_emulators.tiled_eval configs/samudra_llc/eval.yaml \
  --backend cuda \
  --ckpt_path "${CKPT_PATH}" \
  --inference_stride "${INFERENCE_STRIDE}" \
  --inference_time.start "${INFER_START}" \
  --inference_time.end "${INFER_END}" \
  --experiment.name "${EXPERIMENT_NAME}" \
  --experiment.base_output_dir "${BASE_OUTPUT_DIR}" \
  --experiment.data_root "${DATA_ROOT}" \
  --experiment.wandb.mode disabled \
  --experiment.prognostic_vars_key all \
  --experiment.boundary_vars_key all \
  --data.data_location "${DATA_LOCATION}" \
  "${MODEL_ARGS[@]}" \
  "${TILING_ARGS[@]}"

OUT_DIR="${BASE_OUTPUT_DIR}/${EXPERIMENT_NAME}"
RAW_PRED_ZARR="${OUT_DIR}/predictions.zarr"
TARGET_ZARR="${TARGET_ZARR:-${OUT_DIR}/predictions_4d.zarr}"

if [[ "${REPACK}" == "true" ]]; then
  if [[ ! -d "${RAW_PRED_ZARR}" ]]; then
    echo "Expected raw prediction zarr not found: ${RAW_PRED_ZARR}" >&2
    exit 1
  fi
  if [[ -e "${TARGET_ZARR}" && "${REPACK_OVERWRITE}" != "true" ]]; then
    echo "Target zarr already exists: ${TARGET_ZARR}" >&2
    echo "Delete it, set TARGET_ZARR, or set REPACK_OVERWRITE=true." >&2
    exit 1
  fi

  REPACK_ARGS=(
    --input-zarr "${RAW_PRED_ZARR}"
    --output-zarr "${TARGET_ZARR}"
    --fields ${REPACK_FIELDS}
  )
  if [[ "${REPACK_OVERWRITE}" == "true" ]]; then
    REPACK_ARGS+=(--overwrite)
  fi

  echo
  echo "Repacking flat channels into 4D fields: ${TARGET_ZARR}"
  "${PYTHON_BIN}" scripts/repack_flat_prediction_zarr.py "${REPACK_ARGS[@]}"
fi

echo
echo "Done. Canonical stitched predictions: ${RAW_PRED_ZARR}"
if [[ "${REPACK}" == "true" ]]; then
  echo "Repacked 4D fields:                   ${TARGET_ZARR}"
fi
if [[ "${PREBLEND_MODE}" != "none" ]]; then
  echo "Pre-blend disagreement:               ${OUT_DIR}/preblend.zarr"
fi
if [[ "${PERTURBATION}" == "true" ]]; then
  echo "Perturbation response:                ${OUT_DIR}/perturbation.zarr"
fi
