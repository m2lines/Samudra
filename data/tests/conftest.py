# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Bounded PR checks and larger, replayable nightly explorations."""

import os
from pathlib import Path

import pytest

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


def pytest_addoption(parser):
    parser.addoption(
        "--om4-live",
        action="store_true",
        help="run properties of the published OM4 stores, read from OSN",
    )
    parser.addoption(
        "--om4-live-heavy",
        action="store_true",
        help="with --om4-live, also run checks that read years of 1/4 degree data or re-run the pipeline",
    )


def pytest_collection_modifyitems(config, items):
    """Retain exploratory failures in CI without automatically filing them as bugs."""
    tag_project_items(items, project="data", root=Path(__file__).parent)
    live = config.getoption("--om4-live")
    heavy = config.getoption("--om4-live-heavy")
    for item in items:
        if "om4_live" in item.keywords and not live:
            item.add_marker(pytest.mark.skip(reason="needs --om4-live (reads OSN)"))
        elif "om4_live_heavy" in item.keywords and not heavy:
            item.add_marker(
                pytest.mark.skip(reason="needs --om4-live-heavy (run on the DTN)")
            )
