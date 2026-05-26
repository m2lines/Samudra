#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --job-name=2026-05-26-eval:Samudra_LLC:long_curriculum_3-B---3_ckpt-24
#SBATCH --account=mit_amf_standard_gpu
#SBATCH --qos=mit_amf_standard_gpu
#SBATCH -N 1
#SBATCH --mem=100GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1
#SBATCH --time=00-03:00:00
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

CKPT_PATH="${CKPT_PATH:-/home/codycruz/Ocean_Emulator/.LOCAL/2026-05-25:samudra_llc:long_curriculum_3-B---3-14508474/saved_nets/ckpt_24.pt}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${SLURM_JOB_NAME:-$(basename "$0" .sh)}}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-/orcd/data/abodner/002/cody/inference_patch}"
EXPERIMENT_NAME="${EXPERIMENT_NAME}${SLURM_JOB_ID:+-${SLURM_JOB_ID}}"


INFER_START="${INFER_START:-2012-10-14}"
INFER_END="${INFER_END:-2012-10-17}"
INFERENCE_STRIDE="${INFERENCE_STRIDE:-3}"
NUM_MODEL_STEPS_FORWARD="${NUM_MODEL_STEPS_FORWARD:-4}"
MODEL_NORM="${MODEL_NORM:-group}"
GROUP_NORM_GROUPS="${GROUP_NORM_GROUPS:-32}"
PRED_RESIDUALS="${PRED_RESIDUALS:-true}"
MODEL_PAD="${MODEL_PAD:-constant}"
NUM_HALO="${NUM_HALO:-4}"
NUM_SPONGE="${NUM_SPONGE:-12}"

DATA_ROOT="${DATA_ROOT:-/orcd/data/abodner/}"
DATA_LOCATION="${DATA_LOCATION:-/orcd/data/abodner/003/LLC4320/LLC4320}"
LLC_FACE="${LLC_FACE:-1}"
LLC_I_START="${LLC_I_START:-2880}" # 1440 
LLC_I_END="${LLC_I_END:-3600}" #2160 
LLC_J_START="${LLC_J_START:-720}" # 1440
LLC_J_END="${LLC_J_END:-1440}" # 2160


RAW_PRED_ZARR="${RAW_PRED_ZARR:-${BASE_OUTPUT_DIR}/${EXPERIMENT_NAME}/predictions.zarr}"
TARGET_ZARR="${TARGET_ZARR:-${BASE_OUTPUT_DIR}/${EXPERIMENT_NAME}/predictions_4d.zarr}"
REPACK_OVERWRITE="${REPACK_OVERWRITE:-false}"

echo "======== evaluating epoch-1 checkpoint on October 2012 ========"
echo "checkpoint: ${CKPT_PATH}"
echo "inference window: ${INFER_START} -> ${INFER_END}"
echo "inference_stride: ${INFERENCE_STRIDE}"
echo "num_model_steps_forward: ${NUM_MODEL_STEPS_FORWARD}"
if [[ -n "${MODEL_NORM}" ]]; then
  echo "model norm override: ${MODEL_NORM} (group_norm_groups=${GROUP_NORM_GROUPS})"
fi
if [[ -n "${MODEL_PAD}" ]]; then
  echo "model pad override: ${MODEL_PAD} (num_halo=${NUM_HALO}, num_sponge=${NUM_SPONGE})"
fi
if [[ -n "${PRED_RESIDUALS}" ]]; then
  echo "pred_residuals override: ${PRED_RESIDUALS}"
fi
echo "raw prediction zarr: ${RAW_PRED_ZARR}"
echo "target repacked zarr: ${TARGET_ZARR}"
echo "repack overwrite: ${REPACK_OVERWRITE}"
echo "llc crop: face=${LLC_FACE}, i=[${LLC_I_START}:${LLC_I_END}), j=[${LLC_J_START}:${LLC_J_END})"
echo
echo "Note: dates are parsed as Julian-noon in this codebase; with hist=1 this yields"
echo "      prediction times offset from midnight (first prediction starts $((2 * INFERENCE_STRIDE)) hours"
echo "      after the sliced start timestamp)."

MODEL_ARGS=()
if [[ -n "${MODEL_NORM}" ]]; then
  MODEL_ARGS+=(--model.unet.core_block.norm "${MODEL_NORM}")
  if [[ "${MODEL_NORM}" == "group" ]]; then
    MODEL_ARGS+=(--model.unet.core_block.group_norm_groups "${GROUP_NORM_GROUPS}")
  fi
fi
if [[ -n "${MODEL_PAD}" ]]; then
  MODEL_ARGS+=(--model.pad "${MODEL_PAD}")
  if [[ "${MODEL_PAD}" == "halo_sponge" ]]; then
    MODEL_ARGS+=(--model.num_halo "${NUM_HALO}")
    MODEL_ARGS+=(--model.num_sponge "${NUM_SPONGE}")
  fi
fi
if [[ -n "${PRED_RESIDUALS}" ]]; then
  MODEL_ARGS+=(--model.pred_residuals "${PRED_RESIDUALS}")
fi

"${PYTHON_BIN}" -m ocean_emulators.eval configs/samudra_llc/eval.yaml \
  --backend cuda \
  --save_zarr true \
  --ckpt_path "${CKPT_PATH}" \
  --inference_stride "${INFERENCE_STRIDE}" \
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
  "${MODEL_ARGS[@]}" \
  --data.llc_face "${LLC_FACE}" \
  --data.llc_i_start "${LLC_I_START}" \
  --data.llc_i_end "${LLC_I_END}" \
  --data.llc_j_start "${LLC_J_START}" \
  --data.llc_j_end "${LLC_J_END}"

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

"${PYTHON_BIN}" scripts/repack_flat_prediction_zarr.py "${REPACK_ARGS[@]}"

echo "Done. Repacked inference written to: ${TARGET_ZARR}"
echo "Raw flat-channel predictions kept at: ${RAW_PRED_ZARR}"
