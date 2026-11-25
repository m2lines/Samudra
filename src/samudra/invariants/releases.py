# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Two builds of the same data agree, except where a documented change says otherwise.

Use it to compare two releases, or to compare a store rebuilt with the current
pipeline against the published one. A difference nobody documented is either an
unrecorded code change or a regression.
"""

import numpy as np
import xarray as xr

from samudra.invariants.core import Violation, first_location


def stores_agree(
    reference: xr.Dataset,
    candidate: xr.Dataset,
    variables=None,
    rtol: float = 1e-5,
    atol: float = 1e-6,
    expected_changes: frozenset[str] = frozenset(),
) -> list[Violation]:
    """Shared variables agree in shape, NaN pattern and value.

    Args:
        reference: The trusted build, such as the published store.
        candidate: The build being checked, aligned to the same times.
        variables: Variables to compare; defaults to every variable in both.
        rtol: Relative tolerance for values.
        atol: Absolute tolerance for values.
        expected_changes: Variables a documented change allows to differ.
    """
    names = (
        variables
        if variables is not None
        else sorted(map(str, set(reference.variables) & set(candidate.variables)))
    )
    violations = []
    for name in names:
        if name in expected_changes:
            continue
        if name not in reference.variables or name not in candidate.variables:
            violations.append(
                Violation(
                    "releases.variable_present", f"{name} is missing from one build"
                )
            )
            continue
        a, b = reference[name], candidate[name]
        if a.dims != b.dims or a.shape != b.shape:
            violations.append(
                Violation(
                    "releases.same_shape",
                    f"{name}: {dict(a.sizes)} vs {dict(b.sizes)}",
                )
            )
            continue
        if a.dtype.kind not in "fiub" or b.dtype.kind not in "fiub":
            if not np.array_equal(np.asarray(a), np.asarray(b)):
                violations.append(Violation("releases.same_values", f"{name} differs"))
            continue
        # Compare by position: xarray arithmetic would align on coordinates and
        # silently drop cells whose coordinate values differ between builds.
        a_values = np.asarray(a.load())
        b_values = np.asarray(b.load())
        nan_differs = xr.DataArray(
            np.isnan(a_values) != np.isnan(b_values), dims=a.dims
        )
        if nan_differs.any():
            violations.append(
                Violation(
                    "releases.same_nan_pattern",
                    f"{name}: NaN pattern differs in {int(nan_differs.sum())} cells",
                    first_location(nan_differs),
                )
            )
        close = np.isclose(a_values, b_values, rtol=rtol, atol=atol, equal_nan=True)
        far = xr.DataArray(~close, dims=a.dims)
        if far.any():
            worst = float(
                np.nanmax(np.abs(a_values.astype(float) - b_values.astype(float)))
            )
            violations.append(
                Violation(
                    "releases.same_values",
                    f"{name}: {int(far.sum())} values differ; worst by {worst:.4g}",
                    first_location(far),
                )
            )
    return violations
