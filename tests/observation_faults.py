# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Observation-training faults the observation properties must detect.

`python -m samudra_verification.gate tests` finds this file by its `_faults.py` suffix.
"""

from pathlib import Path

from samudra_verification.faults import Catalog, Fault

LOSSES = "tests/test_observation_loss_properties.py"

FAULTS = (
    # Found by the loss property during the #955 soak: with no completion support,
    # the loss summed the raw prediction, so a NaN in any unsupported cell made
    # the loss NaN although no cell was scored.
    Fault(
        "empty_support_reads_prediction",
        "samudra.observations.losses",
        "return error.sum() * 0",
        "return prediction.sum() * 0",
        (
            f"{LOSSES}::"
            "test_masked_losses_ignore_unsupported_cells_and_scale_quadratically",
        ),
    ),
)

CATALOG = Catalog(project=Path(__file__).resolve().parents[1], faults=FAULTS)
