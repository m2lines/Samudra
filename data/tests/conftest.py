# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Bounded PR checks and larger, replayable nightly explorations."""

import os

try:
    from samudra_verification.profiles import configure_profiles, tag_project_items
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "The data tests need samudra-verification. From the repository root, run: "
        "pip install -e ./verification"
    ) from error

if storage := os.environ.get("SAMUDRA_STRING_STORAGE"):
    import pandas as pd

    pd.set_option("mode.string_storage", storage)

configure_profiles()


def pytest_collection_modifyitems(items):
    """Retain exploratory failures in CI without automatically filing them as bugs."""
    tag_project_items(items, project="data")
