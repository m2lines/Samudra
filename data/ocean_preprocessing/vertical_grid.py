# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""OM4 nominal layer centers and thicknesses used by data preparation."""

import numpy as np

OM4_DEPTH_LEVELS = np.array(
    [
        2.5,
        10,
        22.5,
        40,
        65,
        105,
        165,
        250,
        375,
        550,
        775,
        1050,
        1400,
        1850,
        2400,
        3100,
        4000,
        5000,
        6000,
    ],
    dtype=np.float64,
)
# Nominal layers are contiguous, start at 0 m, and have midpoint centers.
# Thus each lower interface is 2 * center - upper interface.
_interfaces = [0.0]
for _center in OM4_DEPTH_LEVELS:
    _interfaces.append(2 * _center - _interfaces[-1])
OM4_LAYER_THICKNESS = np.diff(_interfaces)
