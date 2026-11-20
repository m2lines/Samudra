#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
export SKYLINE_ROOT=/scratch/jr7309/skyline-small-lr-sweep-20261120
export SKYLINE_COMMIT=d06db8c7d5606f7a12de0ee0ead859d86da441c3 # pragma: allowlist secret
export SKYLINE_ARCHIVE_SHA256=a88ed14667afb47c0efccfe96ad881c41f20eef0e17d6b533c5b9e10726bba99 # pragma: allowlist secret
export SKYLINE_MODULE=samudra.skyline.comparison_train
export SKYLINE_WORLD_SIZE=1
export SIF_PATH=/scratch/jr7309/.apptainer-images/ghcr.io-m2lines-ocean-emulator-physicsnemo-26.05-4f86be45a09cb9eba1b51b6273e0189941b02495.sif # pragma: allowlist secret
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE=online
export PYTHONFAULTHANDLER=1
export NCCL_DEBUG=WARN
