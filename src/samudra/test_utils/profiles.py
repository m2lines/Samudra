# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Shared Hypothesis profiles and triage annotations; suite dependencies stay separate."""

import os

from hypothesis import Phase, settings

# The mutation gate checks that each subprocess ran with exactly these settings.
MUTATION_SETTINGS = settings(
    max_examples=40, deadline=None, database=None, phases=[Phase.generate]
)


def profile_evidence(active: settings) -> dict:
    """The settings that decide whether a mutation run is valid, as JSON."""
    return {
        "max_examples": active.max_examples,
        "phases": [phase.name for phase in active.phases],
        "database_disabled": active.database is None,
        "deadline_disabled": active.deadline is None,
    }


def configure_profiles(default_profile="property"):
    """Register the shared profiles and load one.

    `SAMUDRA_HYPOTHESIS_PROFILE` overrides `default_profile`. The profiles are:

    * `unit`: Hypothesis defaults without the per-example deadline, which fails
      spuriously on loaded machines. For suites whose tests set their own budgets.
    * `property`: 40 examples, the bounded PR check.
    * `soak`: `SAMUDRA_HYPOTHESIS_EXAMPLES` examples (default 1,000), for nightly runs.
    * `mutation`: `MUTATION_SETTINGS`, generation only, for the mutation gate.
    """
    settings.register_profile("unit", deadline=None, print_blob=True)
    settings.register_profile(
        "property", max_examples=40, deadline=None, print_blob=True
    )
    settings.register_profile(
        "soak",
        max_examples=int(os.environ.get("SAMUDRA_HYPOTHESIS_EXAMPLES", "1000")),
        deadline=None,
        print_blob=True,
    )
    settings.register_profile("mutation", parent=MUTATION_SETTINGS, print_blob=True)
    settings.load_profile(os.environ.get("SAMUDRA_HYPOTHESIS_PROFILE", default_profile))


def tag_verification_items(items, suite):
    """Keep proposed contracts visible while excluding them from product reports."""
    for item in items:
        item.user_properties.append(("verification_suite", suite))
        if item.get_closest_marker("contract_probe") is not None:
            item.user_properties.append(("contract_probe", "true"))
