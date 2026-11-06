# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Shared profile names and triage annotations; suite dependencies stay separate."""

import os

from hypothesis import Phase, settings


def configure_profiles(default_profile="property"):
    """Opt into a bounded property run or larger soak without changing other suites."""
    settings.register_profile(
        "property", max_examples=40, deadline=None, print_blob=True
    )
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
    selected = os.environ.get("SAMUDRA_HYPOTHESIS_PROFILE", default_profile)
    if selected is not None:
        settings.load_profile(selected)


def tag_verification_items(items, suite):
    """Keep proposed contracts visible while excluding them from product reports."""
    for item in items:
        item.user_properties.append(("verification_suite", suite))
        if item.get_closest_marker("contract_probe") is not None:
            item.user_properties.append(("contract_probe", "true"))
