# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Shared time and channel layout for the surface-initialized observation model."""

STEP_DAYS = 5
HISTORY_STEPS = 19  # Surface/forcing input history: 95 days before initialization.
HISTORY_DAYS = HISTORY_STEPS * STEP_DAYS
STATE_HISTORY_STEPS = 2  # Two physical states advance the autoregressive model.
INITIAL_STATE_START = HISTORY_STEPS - STATE_HISTORY_STEPS
LAST_HISTORY_STEP = HISTORY_STEPS - 1
FORECAST_DAYS = 365
FORECAST_STEPS = FORECAST_DAYS // STEP_DAYS
MAX_REPLAY_MONTHS = 12

MODEL_DEPTHS = 19
OBSERVED_DEPTHS = 14
TEMPERATURE_OFFSET = 2 * MODEL_DEPTHS  # Channel order: U, V, T, S, SSH.
SALINITY_OFFSET = 3 * MODEL_DEPTHS
SSH_CHANNEL = 4 * MODEL_DEPTHS
STATE_CHANNELS = SSH_CHANNEL + 1
SURFACE_CHANNELS = [TEMPERATURE_OFFSET, SSH_CHANNEL]
