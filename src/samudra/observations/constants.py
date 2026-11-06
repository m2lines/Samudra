# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Shared time and channel layout for the surface-initialized observation model."""

from typing import Literal

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

MODEL_VARIABLES = ("uo", "vo", "thetao", "so")
STATE_NAMES = tuple(
    f"{variable}_{depth}"
    for variable in MODEL_VARIABLES
    for depth in range(MODEL_DEPTHS)
) + ("zos",)
SURFACE_NAMES = ("thetao_0", "zos")
PROFILE_VARIABLES = ("thetao", "so")
PROFILE_NAMES = tuple(
    f"{variable}_{depth}"
    for variable in PROFILE_VARIABLES
    for depth in range(OBSERVED_DEPTHS)
)
MODEL_FORCING_VARIABLES = ("tauuo", "tauvo", "hfds")
MODEL_FORCING_CHANNELS = len(MODEL_FORCING_VARIABLES)
OBSERVATION_FORCING_CHANNELS = 8  # ERA5 surface weather fields in prepare.FORCING.
SPHERICAL_COORDINATE_CHANNELS = 3  # Unit-sphere x, y, z.
SEASON_CHANNELS = 2  # Sine and cosine of the native-calendar annual phase.
CONTEXT_CHANNELS = SPHERICAL_COORDINATE_CHANNELS + SEASON_CHANNELS
GEOMETRY_CHANNELS = 7  # Spacing x/y, extent x/y, reserved, wet-any, wet-fraction.
ModelTask = Literal["observation", "om4-family"]
MODEL_TASKS: tuple[ModelTask, ...] = ("om4-family", "observation")
