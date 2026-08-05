# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Colour limits for map panels.

Percentile rather than min/max, so a single outlier cell cannot flatten a whole
figure, with the degenerate cases -- an all-NaN field, a constant one, an
inverted range -- resolved to something matplotlib can draw.

Its own module rather than part of `viz.core` so that `viz.observations` can use
it without the two importing each other.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import xarray as xr
from dask.array.core import Array as DaskArray
from matplotlib import colors

def _flatten_for_norm(data: Any) -> np.ndarray:
    if isinstance(data, xr.DataArray):
        arr = data.data
    elif isinstance(data, np.ndarray):
        arr = data
    elif isinstance(data, DaskArray):
        arr = data
    elif isinstance(data, Iterable) and not isinstance(data, (str, bytes)):
        arrays = [_flatten_for_norm(item) for item in data]
        if arrays:
            return np.concatenate(arrays)
        return np.array([], dtype=float)
    else:
        arr = data

    if isinstance(arr, DaskArray):
        arr = arr.compute()
    return np.asarray(arr).ravel()


def symmetric_percentile_norm(
    data: Any, percentile: float = 98.0, fallback: float = 1.0
) -> colors.Normalize:
    flat = _flatten_for_norm(data)
    flat = flat[~np.isnan(flat)]
    if flat.size == 0:
        max_abs = fallback
    else:
        max_abs = np.percentile(np.abs(flat), percentile)
        if not np.isfinite(max_abs) or max_abs == 0:
            max_abs = fallback
    return colors.Normalize(vmin=-max_abs, vmax=max_abs)


def percentile_norm(
    data: Any, lower: float = 2.0, upper: float = 98.0, fallback: float = 1.0
) -> colors.Normalize:
    flat = _flatten_for_norm(data)
    flat = flat[~np.isnan(flat)]
    if flat.size == 0:
        vmin = 0.0
        vmax = fallback
    else:
        vmin = np.percentile(flat, lower)
        vmax = np.percentile(flat, upper)
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            vmin = 0.0
            vmax = fallback
        elif vmin == vmax:
            vmin = vmin - fallback
            vmax = vmax + fallback
        elif vmin > vmax:
            vmin, vmax = vmax, vmin
    return colors.Normalize(vmin=vmin, vmax=vmax)
