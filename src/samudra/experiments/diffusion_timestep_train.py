# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Matched five-/ten-day endpoint-consistency continuation pilot."""

from samudra.experiments.diffusion_interventions import main

if __name__ == "__main__":
    main(campaign_default="timestep")
