#!/bin/bash
# 1-GPU debug wrapper for the shared Apptainer trainer with pre-train GDS diagnostics.

#SBATCH --job-name=oe-train-gds-debug
#SBATCH --output=slurm-%j.out
#SBATCH --partition=pi_abodner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00

set -euo pipefail

export GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
export PRE_TRAIN_DIAGNOSTICS="${PRE_TRAIN_DIAGNOSTICS:-1}"
export PRE_TRAIN_ZARR_PROBE_PATH="${PRE_TRAIN_ZARR_PROBE_PATH:-/orcd/home/002/jrusak/orcd/pool/data/LLC4320v3}"
export PRE_TRAIN_ZARR_PROBE_VAR="${PRE_TRAIN_ZARR_PROBE_VAR:-Theta}"

if [[ -n "${ARGS:-}" ]]; then
  export ARGS="--debug=true ${ARGS}"
else
  export ARGS="--debug=true"
fi

source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/slurm_apptainer_train.sbatch"
