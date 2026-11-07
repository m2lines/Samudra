#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
set -euo pipefail
: "${ROOT:?}"
cd "$ROOT"
test -f PROBE_PASSED
sha256sum -c inputs.sha256
arms=(legacy_mse wet_mse dynamic_wet)
arm=${arms[$SLURM_ARRAY_TASK_ID]}
export CONFIG="$ROOT/configs/$arm.json" NAME="issue964-$arm-seed15"
export OUTPUT_BASE="$ROOT/outputs"
export DATA_ROOT=/scratch/jr7309/data/om4-v2026-09/om4_onedeg
export DATA_CACHE_DIR="/scratch/jr7309/.data_cache/issue964-$arm"
export REPO_DIR=/scratch/jr7309 SCRATCH_DIR=/scratch/jr7309
export SIF_DIR=/scratch/jr7309/.apptainer-images
export APPTAINER_CACHEDIR=/scratch/jr7309/apptainer-cache
export SINGULARITY_CACHEDIR=/scratch/jr7309/singularity-cache
export WANDB_MODE=online REQUEUE_ON_USR1=1
exec bash "$ROOT/slurm_apptainer_train.sbatch"
