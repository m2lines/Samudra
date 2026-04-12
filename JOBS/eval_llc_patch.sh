#!/bin/bash
#SBATCH -p pi_abodner
#SBATCH --job-name=2026-04-06-eval:samudra_llc:Agulhas,from-4-4-26-RESUME
#SBATCH -N 1
#SBATCH --mem=200GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1
#SBATCH --time=00-06:00:00
#SBATCH -o /orcd/home/002/codycruz/Ocean_Emulator/logs/%x-%j.out
#SBATCH -e /orcd/home/002/codycruz/Ocean_Emulator/logs/%x-%j.out

set -euo pipefail

module load miniforge/24.3.0-0

cd /orcd/home/002/codycruz/Ocean_Emulator
uv sync --dev

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

CKPT_PATH="${CKPT_PATH:-/orcd/home/002/codycruz/Ocean_Emulator/.LOCAL/2026-04-04-CONT:[increase-step-test-suite]-WITH_temporal_stride=6,steps=4,2012-01-01-2012-09-14-RESUME/saved_nets/ckpt.pt}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-2026-04-06-eval:samudra_llc:Agulhas,from-4-4-26-RESUME}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-/orcd/data/abodner/002/cody/inference_patch/predictions}"

INFER_START="${INFER_START:-2012-10-01}"
INFER_END="${INFER_END:-2012-11-01}"
NUM_MODEL_STEPS_FORWARD="${NUM_MODEL_STEPS_FORWARD:-2}"

DATA_ROOT="${DATA_ROOT:-/orcd/data/abodner/}"
DATA_LOCATION="${DATA_LOCATION:-/orcd/data/abodner/002/cody/LLC_patch/LLC4320_face1_i2880-3600_j720-1440.zarr}"

RAW_PRED_ZARR="${RAW_PRED_ZARR:-${BASE_OUTPUT_DIR}/${EXPERIMENT_NAME}/predictions.zarr}"
TARGET_ZARR="${TARGET_ZARR:-${BASE_OUTPUT_DIR}/${EXPERIMENT_NAME}/predictions_4d.zarr}"
REPACK_OVERWRITE="${REPACK_OVERWRITE:-false}"

echo "======== evaluating epoch-1 checkpoint on October 2012 ========"
echo "checkpoint: ${CKPT_PATH}"
echo "inference window: ${INFER_START} -> ${INFER_END}"
echo "num_model_steps_forward: ${NUM_MODEL_STEPS_FORWARD}"
echo "raw prediction zarr: ${RAW_PRED_ZARR}"
echo "target repacked zarr: ${TARGET_ZARR}"
echo "repack overwrite: ${REPACK_OVERWRITE}"
echo
echo "Note: dates are parsed as Julian-noon in this codebase; with hist=1 this yields"
echo "      prediction times offset from midnight (first prediction starts 2 hours"
echo "      after the sliced start timestamp)."

uv run python -m ocean_emulators.eval configs/samudra_llc/eval.yaml \
  --backend cuda \
  --save_zarr true \
  --ckpt_path "${CKPT_PATH}" \
  --num_model_steps_forward "${NUM_MODEL_STEPS_FORWARD}" \
  --inference_time.start "${INFER_START}" \
  --inference_time.end "${INFER_END}" \
  --experiment.name "${EXPERIMENT_NAME}" \
  --experiment.base_output_dir "${BASE_OUTPUT_DIR}" \
  --experiment.data_root "${DATA_ROOT}" \
  --experiment.wandb.mode disabled \
  --experiment.prognostic_vars_key all \
  --experiment.boundary_vars_key all \
  --data.data_location "${DATA_LOCATION}" \
  --data.llc_face 1 \
  --data.llc_i_start 2880 \
  --data.llc_i_end 3600 \
  --data.llc_j_start 720 \
  --data.llc_j_end 1440

if [[ ! -d "${RAW_PRED_ZARR}" ]]; then
  echo "Expected raw prediction zarr not found: ${RAW_PRED_ZARR}" >&2
  exit 1
fi

if [[ -e "${TARGET_ZARR}" && "${REPACK_OVERWRITE}" != "true" ]]; then
  echo "Target zarr already exists: ${TARGET_ZARR}" >&2
  echo "Delete it first, set TARGET_ZARR to a new path, or set REPACK_OVERWRITE=true." >&2
  exit 1
fi

REPACK_ARGS=(
  --input-zarr "${RAW_PRED_ZARR}"
  --output-zarr "${TARGET_ZARR}"
  --fields U V Theta Salt
)
if [[ "${REPACK_OVERWRITE}" == "true" ]]; then
  REPACK_ARGS+=(--overwrite)
fi

uv run python scripts/repack_flat_prediction_zarr.py "${REPACK_ARGS[@]}"

echo "Done. Repacked inference written to: ${TARGET_ZARR}"
echo "Raw flat-channel predictions kept at: ${RAW_PRED_ZARR}"
