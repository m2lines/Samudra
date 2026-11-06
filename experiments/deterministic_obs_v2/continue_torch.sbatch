#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
TASK_ROOT=/scratch/jr7309/deterministic-obs-v2
export NAME="${1:?Pass generated continuation name}"
export CONFIG="$TASK_ROOT/configs/continuation-v1/$NAME.yaml"
export OUTPUT_BASE="$TASK_ROOT/screens-v1" DATA_ROOT=/scratch/jr7309
export SAMUDRA_MODULE=samudra.observations.train SAMUDRA_MODULE_ARGS="$CONFIG"
export SAMUDRA_MANAGE_RUN_DIR=0 GPUS_PER_NODE=4 REQUEUE_ON_USR1=1 WANDB_MODE=online
export SIF_PATH=/scratch/jr7309/.apptainer-images/ghcr.io-m2lines-ocean-emulator-physicsnemo-26.05-4f86be45a09cb9eba1b51b6273e0189941b02495.sif
export CODE_LAYER=/scratch/jr7309/.apptainer-code-layers/samudra-code-7bef723cb1f60d4e202aeece6fd4c1b11eb2e539.img
export REPO_DIR=/scratch/jr7309 SIF_DIR=/scratch/jr7309/.apptainer-images
export APPTAINER_CACHEDIR=/scratch/jr7309/apptainer-cache SINGULARITY_CACHEDIR=/scratch/jr7309/singularity-cache
export DATA_CACHE_DIR="$TASK_ROOT/continuation-v1/data-cache/$NAME"
export NCCL_P2P_DISABLE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export APPTAINERENV_NCCL_P2P_DISABLE=1 APPTAINERENV_TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO APPTAINERENV_NCCL_DEBUG=INFO
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$OUTPUT_BASE/telemetry"
printf '%s\n' "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:?}" > "$OUTPUT_BASE/telemetry/${SLURM_JOB_ID}-${SLURM_RESTART_COUNT:-0}.devices"
nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,uuid,utilization.gpu,memory.used --format=csv --loop=5 > "$OUTPUT_BASE/telemetry/${SLURM_JOB_ID}-${SLURM_RESTART_COUNT:-0}.csv" &
exec bash "$TASK_ROOT/frozen-data/train-scaling-nccl-25af85a34ff8.sbatch"
