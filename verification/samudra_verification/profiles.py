# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Shared Hypothesis profiles, and the tags each project's conftest puts on its tests."""

import os
from dataclasses import dataclass
from pathlib import Path

import pytest
from hypothesis import Phase, register_random, settings

# Hypothesis keeps only weak references to registered generators.
_REGISTERED_GENERATORS: list["TorchRandom"] = []


class TorchRandom:
    """torch's global generator, in the `random.Random` shape Hypothesis seeds."""

    def __init__(self, torch):
        self.seed = torch.manual_seed
        self.getstate = torch.get_rng_state
        self.setstate = torch.set_rng_state


def register_torch_random():
    """Let Hypothesis seed torch's global generator, as it does `random` and NumPy's.

    Otherwise a property that draws from torch (initialization, dropout, `randn`)
    sees different values when Hypothesis replays an example, so shrinking reports
    `Flaky` and the mutation gate marks the trial invalid. `torch.manual_seed`
    seeds every device; only the CPU generator's state is saved and restored, and
    property runs are CPU-only. Call it from the conftest of any project whose
    properties use torch. This module doesn't import torch otherwise.
    """
    import torch

    if not _REGISTERED_GENERATORS:
        generator = TorchRandom(torch)
        _REGISTERED_GENERATORS.append(generator)
        register_random(generator)


# The mutation gate checks that each subprocess ran with exactly these settings.
# Explicit @examples run because they are often a fault's known reproducer; the
# database and shrinking stay off so each trial depends only on its seed.
MUTATION_SETTINGS = settings(
    max_examples=40,
    deadline=None,
    database=None,
    phases=[Phase.explicit, Phase.generate],
)


@dataclass(frozen=True)
class Profile:
    """The settings that decide whether a mutation trial is valid."""

    max_examples: int
    phases: tuple[str, ...]
    database_disabled: bool
    deadline_disabled: bool

    @classmethod
    def of(cls, active: settings) -> "Profile":
        return cls(
            max_examples=active.max_examples,
            phases=tuple(phase.name for phase in active.phases),
            database_disabled=active.database is None,
            deadline_disabled=active.deadline is None,
        )


def configure_profiles(default_profile="property"):
    """Register the shared profiles and load one.

    `SAMUDRA_HYPOTHESIS_PROFILE` overrides `default_profile`. The profiles are:

    * `unit`: Hypothesis defaults without the per-example deadline, which fails
      spuriously on loaded machines. For suites whose tests set their own budgets.
    * `property`: 40 examples, the bounded PR check.
    * `soak`: `SAMUDRA_HYPOTHESIS_EXAMPLES` examples (default 1,000), for nightly runs.
    * `mutation`: `MUTATION_SETTINGS`, explicit and generated examples without
      shrinking or the database, for the mutation gate.
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


INVARIANT_STATUSES = ("proposed", "agreed")


def tag_project_items(items, project, root):
    """Tag each test under `root` with its project, and its invariant's status if marked.

    Pytest calls every conftest's `pytest_collection_modifyitems` with every
    collected item, so each project passes its own test directory as `root`.

    Mark a property whose invariant the team hasn't agreed on yet with
    `@pytest.mark.invariant(status="proposed")`. Its failures still appear in
    results, but triage won't propose them as bug reports.
    """
    root = Path(root).resolve()
    for item in items:
        if not item.path.resolve().is_relative_to(root):
            continue
        item.user_properties.append(("project", project))
        if (marker := item.get_closest_marker("invariant")) is not None:
            status = marker.kwargs.get("status")
            if status not in INVARIANT_STATUSES:
                raise pytest.UsageError(
                    f"{item.nodeid}: write the marker as "
                    '@pytest.mark.invariant(status="proposed") or (status="agreed"), '
                    f"not with args={marker.args!r}, kwargs={marker.kwargs!r}"
                )
            item.user_properties.append(("invariant_status", status))
