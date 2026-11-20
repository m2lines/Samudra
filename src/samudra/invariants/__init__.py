# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Ocean-physics invariants as pure functions over xarray objects.

Each check takes data and returns a list of `Violation`s; an empty list means the
data satisfies the invariant. Nothing here writes, plots, or imports torch, so
the data pipeline's property tests, the model's tests, and audits of published
stores can share these checks. `docs/concepts/ocean-physics.md` explains the
physics behind each one.

Checks fall into two groups:

* Single-dataset checks (`masks`, `bounds`, `geometry`, `timeaxis`, `budgets`)
  hold for any one correct dataset.
* Metamorphic checks (`conservation`, `coupling`, `releases`) compare two versions of the same
  ocean, such as one field at two resolutions, before and after a pipeline step,
  or two releases, and check the relation physics says must hold between them.
"""

from samudra.invariants.core import InvariantViolated, Violation, require_no_violations

__all__ = ["InvariantViolated", "Violation", "require_no_violations"]
