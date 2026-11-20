#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
export SKYLINE_ROOT=/scratch/jr7309/skyline-small-lr-sweep-20261120
export SKYLINE_COMMIT=b7abb23f54e1be8b569cbf237ea6aff9660c0ab7 # pragma: allowlist secret
export SKYLINE_ARCHIVE_SHA256=9ce1f2c4ae20a706b3895e16c83889c814ff127d1904b516833478fff0541818 # pragma: allowlist secret
export SKYLINE_MODULE=samudra.skyline.comparison_train
export SKYLINE_WORLD_SIZE=1
export SIF_PATH=/scratch/jr7309/.apptainer-images/ghcr.io-m2lines-ocean-emulator-physicsnemo-26.05-4f86be45a09cb9eba1b51b6273e0189941b02495.sif # pragma: allowlist secret
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE=online
export PYTHONFAULTHANDLER=1
export NCCL_DEBUG=WARN
