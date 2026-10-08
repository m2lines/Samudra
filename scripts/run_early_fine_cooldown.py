#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Five-arm cooldown wrapper, including the unchanged recent-only control."""

from run_early_fine_wave import ARMS, main

if __name__ == "__main__":
    main(dict(ARMS, **{"U-global": ("unet", 0)}))
