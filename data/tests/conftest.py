# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Bounded PR checks and larger, replayable nightly explorations."""

import importlib.util
import os
from pathlib import Path

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

# tests/virtual covers ocean_preprocessing/virtualize_om4.py, which runs in its own
# environment (zarr-python 3, VirtualiZarr; see the script's header). The CI jobs
# that build that environment run it; the main data environment can't import it.
collect_ignore = [] if importlib.util.find_spec("virtualizarr") else ["virtual"]


def pytest_collection_modifyitems(items):
    """Retain exploratory failures in CI without automatically filing them as bugs."""
    tag_project_items(items, project="data", root=Path(__file__).parent)
