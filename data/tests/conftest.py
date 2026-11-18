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


def _in_virtualize_environment() -> bool:
    """Whether this is virtualize_om4.py's own environment: VirtualiZarr on zarr-python 3.

    An environment updated from an older mamba_env.yaml can still hold VirtualiZarr
    next to zarr 2, where it can't import, so check zarr's version too.
    """
    if importlib.util.find_spec("virtualizarr") is None:
        return False
    import zarr

    return int(zarr.__version__.split(".")[0]) >= 3


# tests/virtual covers ocean_preprocessing/virtualize_om4.py, which runs in its own
# environment (the script's PEP 723 header). CI doesn't run them; data/AGENTS.md has
# the command that does. Everywhere else, they aren't collected.
collect_ignore = [] if _in_virtualize_environment() else ["virtual"]


def pytest_collection_modifyitems(items):
    """Retain exploratory failures in CI without automatically filing them as bugs."""
    tag_project_items(items, project="data", root=Path(__file__).parent)
