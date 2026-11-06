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
export DATA_CACHE_DIR="$TASK_ROOT/scaling-v4/data-cache"
export DATA_ROOT=/scratch/jr7309 OUTPUT_BASE="$TASK_ROOT/scaling-v4"
export WANDB_MODE=online REQUEUE_ON_USR1=1 SAMUDRA_MANAGE_RUN_DIR=0
export SAMUDRA_MODULE=samudra.observations.qualify_scaling
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
for GPUS_PER_NODE in 4; do
  export GPUS_PER_NODE
  export NAME="D-mixed-large-${GPUS_PER_NODE}gpu"
  export CONFIG="$TASK_ROOT/configs/scaling-v4/${NAME}.yaml"
  export SAMUDRA_MODULE_ARGS="$CONFIG"
  memory=128G
  if [[ "$GPUS_PER_NODE" == 1 ]]; then memory=96G; fi
  request=(--chdir=/scratch/jr7309 --account=torch_pr_347_lzanna --nodes=1 --ntasks=1 --cpus-per-task=$((8 * GPUS_PER_NODE)) --mem="$memory" --gres="gpu:$GPUS_PER_NODE" --constraint=rtx6000 --time=01:30:00 --requeue --signal=B:USR1@300 '--comment=preemption=yes;requeue=true' --job-name="obs-scale-v4-${GPUS_PER_NODE}" --output="$TASK_ROOT/scaling-v4-${GPUS_PER_NODE}gpu-%j.out" --error="$TASK_ROOT/scaling-v4-${GPUS_PER_NODE}gpu-%j.out")
  if [[ "$GPUS_PER_NODE" == 1 ]]; then
    request+=(--dependency=afterok:21816667)
  fi
  if [[ "$GPUS_PER_NODE" == 8 && -n "${single_gpu_job:-}" ]]; then
    request+=(--dependency="afterok:${single_gpu_job}")
  fi
  sbatch --test-only "${request[@]}" "/scratch/jr7309/deterministic-obs-v2/frozen-data/train-scaling-nccl-25af85a34ff8.sbatch"
  if [[ "${1:-}" == submit ]]; then
    submitted=$(sbatch --parsable "${request[@]}" "/scratch/jr7309/deterministic-obs-v2/frozen-data/train-scaling-nccl-25af85a34ff8.sbatch")
    echo "${GPUS_PER_NODE} GPUs: ${submitted}"
    if [[ "$GPUS_PER_NODE" == 1 ]]; then single_gpu_job="${submitted%%;*}"; fi
  fi
done
