#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=obs-v2-qualify
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
set -euo pipefail
TASK_ROOT=/scratch/jr7309/deterministic-obs-v2
TASK_ARMS=(A-obs-small B-mixed-small C-obs-large D-mixed-large E-mixed-large-no-replay)
TASK_ARM="${TASK_ARMS[${SLURM_ARRAY_TASK_ID:-0}]}"
test -f "$TASK_ROOT/PREPARED.json"
export CONFIG="$TASK_ROOT/qualification-configs/${TASK_ARM}.yaml"
export NAME="$TASK_ARM"
export REPO_DIR=/scratch/jr7309
export SCRATCH_DIR=/scratch/jr7309
export DATA_ROOT=/scratch/jr7309
export OUTPUT_BASE="$TASK_ROOT/qualification"
export SIF_DIR=/scratch/jr7309/.apptainer-images
export CONTAINER_HASH=4f86be45a09cb9eba1b51b6273e0189941b02495
export CODE_LAYER=/scratch/jr7309/.apptainer-code-layers/samudra-code-acecbf86c9032fb3e13c25886b9d6c1a24acbc40.img
export APPTAINER_CACHEDIR=/scratch/jr7309/apptainer-cache
export SINGULARITY_CACHEDIR=/scratch/jr7309/singularity-cache
export DATA_CACHE_DIR="/scratch/jr7309/.data_cache/obs-v2-${TASK_ARM}"
export SAMUDRA_MODULE=samudra.observations.qualify
export SAMUDRA_MODULE_ARGS="$CONFIG"
export SAMUDRA_MANAGE_RUN_DIR=0
export REQUEUE_ON_USR1=1
export GPUS_PER_NODE=1
export WANDB_MODE=online
mkdir -p "$OUTPUT_BASE" "$DATA_CACHE_DIR"
exec bash "$TASK_ROOT/train.sbatch"
