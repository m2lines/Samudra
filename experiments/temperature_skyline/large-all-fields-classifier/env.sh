#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
export SKYLINE_ROOT=/scratch/jr7309/skyline-large-classifier-20261112
export SKYLINE_COMMIT=ea592e8a71ebf9b50285ef835d641aae1de4d3c9 # pragma: allowlist secret
export SKYLINE_ARCHIVE_SHA256=c9a0e0ffa7aaa3dac98286b1fc5438e0ffaa825f842bbb4e2a423a6434a0d0c3 # pragma: allowlist secret
export SKYLINE_MODULE=samudra.skyline.comparison_train
export SKYLINE_WORLD_SIZE=4
export SIF_PATH=/scratch/jr7309/.apptainer-images/ghcr.io-m2lines-ocean-emulator-physicsnemo-26.05-4f86be45a09cb9eba1b51b6273e0189941b02495.sif # pragma: allowlist secret
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE=online
