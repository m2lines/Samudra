#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
export SKYLINE_ROOT=/scratch/jr7309/skyline-large-high-lr-20261120
export SKYLINE_COMMIT=051c647598da10b82739963b04020e7a0c1d1c1c # pragma: allowlist secret
export SKYLINE_ARCHIVE_SHA256=4acb3aec07cf94bee84db549d1015d6f6314697a03aaf4ceac2c854f063e7484 # pragma: allowlist secret
export SKYLINE_MODULE=samudra.skyline.comparison_train
export SKYLINE_WORLD_SIZE=4
export SIF_PATH=/scratch/jr7309/.apptainer-images/ghcr.io-m2lines-ocean-emulator-physicsnemo-26.05-4f86be45a09cb9eba1b51b6273e0189941b02495.sif # pragma: allowlist secret
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE=online
export PYTHONFAULTHANDLER=1
export NCCL_DEBUG=WARN
