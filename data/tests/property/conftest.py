# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Bounded PR checks and larger, replayable nightly explorations."""

import os

from hypothesis import Phase, settings

if storage := os.environ.get("SAMUDRA_STRING_STORAGE"):
    import pandas as pd

    pd.set_option("mode.string_storage", storage)

settings.register_profile("property", max_examples=40, deadline=None, print_blob=True)
settings.register_profile(
    "soak",
    max_examples=int(os.environ.get("SAMUDRA_HYPOTHESIS_EXAMPLES", "1000")),
    deadline=None,
    print_blob=True,
)
settings.register_profile(
    "mutation",
    max_examples=40,
    deadline=None,
    database=None,
    phases=[Phase.generate],
    print_blob=True,
)
settings.load_profile(os.environ.get("SAMUDRA_HYPOTHESIS_PROFILE", "property"))


def pytest_collection_modifyitems(items):
    """Retain exploratory failures in CI without automatically filing them as bugs."""
    for item in items:
        if item.get_closest_marker("contract_probe") is not None:
            item.user_properties.append(("contract_probe", "true"))
