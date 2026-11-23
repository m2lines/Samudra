# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""The violation record every invariant returns, and helpers shared by the checks."""

import dataclasses
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import xarray as xr

SECONDS_PER_DAY = 86_400.0


@dataclasses.dataclass(frozen=True)
class Violation:
    """One failure of one invariant.

    Attributes:
        invariant: The check's name, e.g. ``"masks.nest_with_depth"``.
        message: What failed, with the numbers needed to judge it.
        where: Coordinates of an example location (e.g. ``{"lev": 3, "y": 10}``),
            or details of an aggregate failure. Empty when not applicable.
    """

    invariant: str
    message: str
    where: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __str__(self) -> str:
        location = f" at {dict(self.where)}" if self.where else ""
        return f"{self.invariant}: {self.message}{location}"


class InvariantViolated(AssertionError):
    """Raised by `require_no_violations`. An `AssertionError`, so tests treat it as an oracle failure."""


def require_no_violations(violations: Iterable[Violation]) -> None:
    """Raise `InvariantViolated` listing every violation, if there are any."""
    found = list(violations)
    if found:
        shown = "\n".join(f"  - {violation}" for violation in found[:20])
        more = f"\n  ... and {len(found) - 20} more" if len(found) > 20 else ""
        raise InvariantViolated(f"{len(found)} invariant violation(s):\n{shown}{more}")


def first_location(flags: xr.DataArray) -> dict[str, Any]:
    """Index of the first True element of a boolean array, keyed by dimension name."""
    values = np.asarray(flags.values)
    if not values.any():
        return {}
    index = np.unravel_index(int(np.argmax(values)), values.shape)
    return {str(dim): int(i) for dim, i in zip(flags.dims, index)}


def isfinite(field: xr.DataArray) -> xr.DataArray:
    """Elementwise ``np.isfinite``, keeping the DataArray's dimensions."""
    return xr.apply_ufunc(np.isfinite, field, dask="allowed")


def same_order(first: xr.DataArray, *others: xr.DataArray) -> list[xr.DataArray]:
    """``others`` transposed to ``first``'s dimension order, so they can be compared position by position.

    Raises:
        ValueError: if any has different dimensions from ``first``.
    """
    for other in others:
        if set(other.dims) != set(first.dims):
            raise ValueError(
                f"cannot compare totals over {first.dims} with totals over {other.dims}"
            )
    return [first, *(other.transpose(*first.dims) for other in others)]


def relative_difference(a: float, b: float) -> float:
    """``|a - b| / max(|a|, |b|)``, or 0 when both are 0."""
    scale = max(abs(a), abs(b))
    return 0.0 if scale == 0 else abs(a - b) / scale


def weighted_sum(field: xr.DataArray, weights: xr.DataArray, dims) -> xr.DataArray:
    """Sum of ``field * weights`` over ``dims``, skipping cells where either is NaN.

    NaN where no cell is valid, so a comparison of an all-NaN field fails instead
    of passing on 0 == 0.
    """
    product = (field * weights).where(field.notnull() & weights.notnull())
    return product.sum(dims, skipna=True, min_count=1)


def weighted_mean(field: xr.DataArray, weights: xr.DataArray, dims) -> xr.DataArray:
    """Mean of ``field`` weighted by ``weights`` over ``dims``, over cells where both are defined."""
    valid = field.notnull() & weights.notnull()
    total_weight = weights.where(valid).sum(dims, skipna=True)
    return weighted_sum(field, weights, dims) / total_weight
