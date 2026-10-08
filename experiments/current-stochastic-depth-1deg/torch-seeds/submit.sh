#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
root=/scratch/jr7309/runs/current-sd-seeds16-17
test -f "$root/DATA_READY"
test -f "$root/CONTAINER_READY"
export REPO_DIR=/scratch/jr7309 SCRATCH_DIR=/scratch/jr7309
export SIF_DIR=/scratch/jr7309/.apptainer-images
export SIF_PATH="$SIF_DIR/sd-seeds-base-0b5320248.sif"
export IMAGE_REF=ghcr.io/m2lines/ocean-emulator-physicsnemo@sha256:babf82575aed4e7cf0166313424e0203b5ea16e09cd525a0bad2874b808d676d
export CODE_LAYER=/scratch/jr7309/.apptainer-code-layers/samudra-code-c3df51b5e0a8e497201377b27423eaf91d712c3e.img
export DATA_ROOT=/scratch/jr7309/data/om4-v2026-09/om4_onedeg
export OUTPUT_BASE="$root/output"
export APPTAINER_CACHEDIR=/scratch/jr7309/apptainer-cache
export SINGULARITY_CACHEDIR=/scratch/jr7309/singularity-cache
export WANDB_MODE=online
export NCCL_P2P_DISABLE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
mkdir -p "$OUTPUT_BASE"
for seed in 16 17; do
  for variant in control sd01; do
    export NAME="current-1deg-${variant}-seed${seed}-torch"
    export NAME_SUFFIX="$NAME"
    export CONFIG="$root/configs/${variant}-seed${seed}.yaml"
    export DATA_CACHE_DIR="/scratch/jr7309/.data_cache/$NAME"
    test ! -e "$root/${variant}-seed${seed}.jobid"
    mkdir -p "$DATA_CACHE_DIR"
    sbatch --parsable --account=torch_pr_347_lzanna --partition=rtx6000_lzanna \
      --nodes=1 --ntasks-per-node=1 --cpus-per-task=128 --gres=gpu:rtx6000:8 \
      --exclusive --mem=0 --time=48:00:00 --job-name="$NAME" \
      --chdir=/scratch/jr7309 --output="$root/train-%j.out" --error="$root/train-%j.err" \
      "$HOME/slurm_apptainer_train.sbatch" | tee "$root/${variant}-seed${seed}.jobid"
  done
done
