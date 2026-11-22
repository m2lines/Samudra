#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
export SKYLINE_ROOT=/scratch/jr7309/skyline-large-classifier-20261112
export SKYLINE_COMMIT=7cf57f71ef52cc1ca9e715958ca740a8bd2b80d2 # pragma: allowlist secret
export SKYLINE_ARCHIVE_SHA256=c675de7b99e475be79fdaa7f1919b3a0cec6e3debe2ef2a0d54ccc9db39e0ac4 # pragma: allowlist secret
export SKYLINE_MODULE=samudra.skyline.comparison_train
export SKYLINE_WORLD_SIZE=4
export SIF_PATH=/scratch/jr7309/.apptainer-images/ghcr.io-m2lines-ocean-emulator-physicsnemo-26.05-4f86be45a09cb9eba1b51b6273e0189941b02495.sif # pragma: allowlist secret
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE=online
export PYTHONFAULTHANDLER=1
export NCCL_DEBUG=WARN
