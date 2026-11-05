#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
export NCCL_P2P_DISABLE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export APPTAINERENV_NCCL_P2P_DISABLE=1 APPTAINERENV_TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO APPTAINERENV_NCCL_DEBUG=INFO
TASK_ROOT=/scratch/jr7309/deterministic-obs-v2
export SIF_PATH=/scratch/jr7309/.apptainer-images/ghcr.io-m2lines-ocean-emulator-physicsnemo-26.05-4f86be45a09cb9eba1b51b6273e0189941b02495.sif
export CODE_LAYER=/scratch/jr7309/.apptainer-code-layers/samudra-code-7bef723cb1f60d4e202aeece6fd4c1b11eb2e539.img
export REPO_DIR=/scratch/jr7309 SIF_DIR=/scratch/jr7309/.apptainer-images
export APPTAINER_CACHEDIR=/scratch/jr7309/apptainer-cache SINGULARITY_CACHEDIR=/scratch/jr7309/singularity-cache
export DATA_CACHE_DIR="$TASK_ROOT/scheduler-v1-4gpu/data-cache"
export DATA_ROOT=/scratch/jr7309 OUTPUT_BASE="$TASK_ROOT/scheduler-v1-4gpu"
export WANDB_MODE=online REQUEUE_ON_USR1=1 SAMUDRA_MANAGE_RUN_DIR=0
export SAMUDRA_MODULE=samudra.observations.train
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export GPUS_PER_NODE=4
export CONFIG="$TASK_ROOT/configs/scheduler-v1-4gpu.yaml"
export TRAIN_HARNESS="/scratch/jr7309/deterministic-obs-v2/frozen-data/train-scaling-nccl-25af85a34ff8.sbatch"
export RESUME_AUDIT_SCRIPT="$TASK_ROOT/frozen-data/scheduler_resume-5bfcce5e4.py"
mkdir -p "$OUTPUT_BASE"
request=(--chdir=/scratch/jr7309 --account=torch_pr_347_lzanna --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=128G --gres=gpu:4 --constraint=rtx6000 --time=01:00:00 --requeue --signal=B:USR1@300 '--comment=preemption=yes;requeue=true' --dependency="afterok:${QUALIFICATION_JOB:?Set the successful qualification dependency}" --kill-on-invalid-dep=yes --output="$TASK_ROOT/scheduler-v1-4gpu-%j.out" --error="$TASK_ROOT/scheduler-v1-4gpu-%j.out")
sbatch --test-only "${request[@]}" "$TASK_ROOT/frozen-data/scheduler_resume-5bfcce5e4.sbatch"
if [[ "${1:-}" == submit ]]; then
  sbatch --parsable "${request[@]}" "$TASK_ROOT/frozen-data/scheduler_resume-5bfcce5e4.sbatch"
fi
