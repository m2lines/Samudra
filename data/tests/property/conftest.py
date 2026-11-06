# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Bounded PR checks and larger, replayable nightly explorations."""

import os

from hypothesis import Phase, settings

from scripts.property_testing import configure_profiles, tag_verification_items

if storage := os.environ.get("SAMUDRA_STRING_STORAGE"):
    import pandas as pd

    pd.set_option("mode.string_storage", storage)

settings.register_profile(
    "mutation",
    max_examples=40,
    deadline=None,
    database=None,
    phases=[Phase.generate],
    print_blob=True,
)
configure_profiles()


def pytest_collection_modifyitems(items):
    """Retain exploratory failures in CI without automatically filing them as bugs."""
    tag_verification_items(items, suite="data")
