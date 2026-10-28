# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Legacy OM4 layer geometry, shared by legacy repair and observation preparation."""

import numpy as np

OM4_LAYER_THICKNESS = np.array(
    [
        5,
        10,
        15,
        20,
        30,
        50,
        70,
        100,
        150,
        200,
        250,
        300,
        400,
        500,
        600,
        800,
        1000,
        1000,
        1000,
    ],
    dtype=np.float64,
)
OM4_DEPTH_LEVELS = np.cumsum(OM4_LAYER_THICKNESS) - OM4_LAYER_THICKNESS / 2
