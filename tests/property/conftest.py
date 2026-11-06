# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""CPU properties are discovered by ordinary PR tests and the project-wide soak."""

from scripts.property_testing import configure_profiles, tag_verification_items

# Preserve ordinary main-project Hypothesis defaults unless a profile is selected.
configure_profiles(default_profile=None)


def pytest_collection_modifyitems(items):
    tag_verification_items(items, suite="samudra")
