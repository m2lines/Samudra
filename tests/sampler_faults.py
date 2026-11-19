# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Sampler faults the sampler properties must detect.

`python -m samudra_verification.gate tests` finds this file by its `_faults.py` suffix.
"""

from pathlib import Path

from samudra_verification.faults import Catalog, Fault

PROPERTIES = "tests/test_sampler_properties.py"

FAULTS = (
    # #778: shuffling wasn't seeded, so a resumed run saw a different batch order
    # than an uninterrupted one. It was fixed in the single-process sampler; this
    # plants the same bug in the distributed one.
    Fault(
        "unseeded_shuffle",
        "samudra.utils.samplers",
        "identical across ranks.\n        rng = random.Random(self.seed + self.epoch)",
        "identical across ranks.\n        rng = random.Random()",
        (f"{PROPERTIES}::test_epoch_replay_is_independent_of_previous_iteration",),
    ),
)

CATALOG = Catalog(project=Path(__file__).resolve().parents[1], faults=FAULTS)
