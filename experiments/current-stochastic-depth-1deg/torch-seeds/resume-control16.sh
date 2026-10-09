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
export NAME=current-1deg-control-seed16-torch-retry1
export NAME_SUFFIX="$NAME"
export CONFIG="$root/configs/control-seed16.yaml"
export DATA_CACHE_DIR="/scratch/jr7309/.data_cache/$NAME"
export ARGS="--resume_ckpt_path=$root/output/current-1deg-control-seed16-torch/saved_nets/ckpt.pt"
test -s "$root/output/current-1deg-control-seed16-torch/saved_nets/ckpt.pt"
sbatch --parsable --account=torch_pr_347_lzanna --partition=rtx6000_lzanna \
 --nodes=1 --ntasks-per-node=1 --cpus-per-task=128 --gres=gpu:rtx6000:8 \
 --mem=512G --time=48:00:00 --job-name="$NAME" --chdir=/scratch/jr7309 \
 --output="$root/train-%j.out" --error="$root/train-%j.err" \
 "$HOME/slurm_apptainer_train.sbatch" | tee "$root/control-seed16-retry1.jobid"
