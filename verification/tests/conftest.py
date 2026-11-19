# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Properties of samudra_verification itself, run in the root environment."""

from pathlib import Path

from samudra_verification.profiles import configure_profiles, tag_project_items

configure_profiles(default_profile="unit")  # As tests/conftest.py, which runs with it.


def pytest_collection_modifyitems(items):
    tag_project_items(items, project="verification", root=Path(__file__).parent)
