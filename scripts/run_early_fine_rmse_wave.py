#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Seven fresh fixed-rate runs with explicit RMSE-only selection."""

from scripts.run_early_fine_wave import ARMS as EARLY_ARMS
from scripts.run_early_fine_wave import main

ARMS = dict(
    EARLY_ARMS,
    **{
        "U-global": ("unet", 0),
        "U-global-seed1730": ("unet", 0),
        "U-global-repro4k": ("unet", 0),
    },
)

if __name__ == "__main__":
    main(ARMS)
