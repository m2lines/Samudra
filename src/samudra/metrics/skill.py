# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Skill against reference forecasts: pure kernels for the scorecard.

A raw error says little on its own: deep-ocean RMSE looks excellent even for a
model that has learned nothing, because the deep ocean barely changes. So the
scorecard always asks how a model's error compares with the error of a
*reference forecast* scored on the same cells and timestamps:

- **persistence**: the initial state, held fixed. Hard to beat at short leads.
- **climatology**: the pentad-of-year mean over the training years. Hard to
  beat at long leads, where the initial state is forgotten.

The skill score is ``1 - MSE_model / MSE_reference``: 1 is perfect, 0 is no
better than the reference, and negative is worse.

Everything is built from **sums**, not means. For every channel and timestamp,
`time_sums` reduces the fields to a handful of area-weighted sums (squared
errors of the model and of each reference, the signed error, the weight). Sums
add, so a calendar year, a depth layer, or the whole window is a sum of rows,
and a ratio of sums is taken only at the end. That keeps one definition behind
every number in the scorecard, and keeps the expensive part (reading the
fields) a single pass whose output is tiny.

Conventions match `samudra.metrics.kernels`: the error is *model minus truth*,
and spatial reductions are weighted by cell area over cells where the truth is
finite.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import xarray as xr

from samudra.metrics import kernels

SPATIAL_DIMS = ("lat", "lon")

# Columns `time_sums` produces for each reference, given its name.
SE_PREFIX = "se_"


def pentad_climatology(truth: xr.DataArray, time_dim: str = "time") -> xr.DataArray:
    """Mean of `truth` in each 5-day bin of the year, over all its years.

    Bins follow `kernels.pentad_index` (a 365-day calendar, February 29 joins
    February 28), so the result has a `pentad` dim of up to 73 bins.

    Raises:
        ValueError: if some bin has no samples, which would leave the
            climatology undefined on those dates.
    """
    index = kernels.coerce_datetime_values(truth[time_dim].values)
    pentads = xr.DataArray(kernels.pentad_index(index), dims=time_dim, name="pentad")
    clim = truth.groupby(pentads).mean(time_dim)
    missing = sorted(set(range(1, 74)) - set(int(p) for p in clim["pentad"].values))
    if missing:
        raise ValueError(
            f"Climatology has no samples for pentad(s) {missing}; give it at least "
            "one full year of 5-day data."
        )
    return clim


def climatology_forecast(
    clim: xr.DataArray, times: xr.DataArray, time_dim: str = "time"
) -> xr.DataArray:
    """The climatology as a forecast: each timestamp gets its pentad's mean."""
    index = kernels.coerce_datetime_values(times.values)
    pentads = xr.DataArray(kernels.pentad_index(index), dims=time_dim)
    out = clim.sel(pentad=pentads).drop_vars("pentad")
    return out.assign_coords({time_dim: times.values})


def time_sums(
    forecast: xr.DataArray,
    truth: xr.DataArray,
    references: dict[str, xr.DataArray],
    area: xr.DataArray,
    spatial_dims: tuple[str, str] = SPATIAL_DIMS,
) -> xr.Dataset:
    """Per-timestamp area-weighted sums for one channel. The scorecard's one pass.

    Cells count where the truth is finite (the wet cells at this level). On
    those cells:

    - ``weight``: total area, Σ w.
    - ``se_model``: Σ w (forecast - truth)².
    - ``se_<name>``: Σ w (reference - truth)², for each reference.
    - ``error``: Σ w (forecast - truth), so ``error / weight`` is the mean bias.
    - ``nonfinite``: area of wet cells where the forecast is not finite. A
      healthy forecast has 0 everywhere; when it is not, ``se_model`` is not
      finite for that timestamp, so the failure cannot hide in an average.

    Args:
        forecast: The model's field, with a time dim and the two spatial dims.
        truth: The truth on the same grid and timestamps.
        references: Reference forecasts on the same grid and timestamps.
        area: Cell area on the same grid.
        spatial_dims: The two horizontal dims to reduce over.

    Returns:
        A dataset of 1-D variables over time. Lazy if the inputs are.

    Raises:
        ValueError: if timestamps or grids differ. Nothing is interpolated.
    """
    kernels.require_exact_time_match(forecast, truth, "scorecard forecast vs truth")
    for name, ref in references.items():
        kernels.require_exact_time_match(ref, truth, f"scorecard reference {name!r}")
        if name == "model":
            raise ValueError("'model' is reserved for the forecast being scored")
    for name, field in {"forecast": forecast, **references, "area": area}.items():
        for dim in spatial_dims:
            if field.sizes.get(dim) != truth.sizes[dim]:
                raise ValueError(
                    f"{name} has {field.sizes.get(dim)} cells along {dim!r}; "
                    f"the truth has {truth.sizes[dim]}. Regrid before scoring."
                )

    wet = cast(xr.DataArray, np.isfinite(truth))
    w = area.where(wet, 0.0)
    out = {
        "weight": w.sum(spatial_dims),
        "nonfinite": w.where(~np.isfinite(forecast), 0.0).sum(spatial_dims),
    }
    for name, field in {"model": forecast, **references}.items():
        err = (field - truth).where(wet, 0.0)
        # skipna=False: a non-finite forecast on a wet cell poisons its sum.
        out[f"{SE_PREFIX}{name}"] = (w * err**2).sum(spatial_dims, skipna=False)
    out["error"] = (w * (forecast - truth).where(wet, 0.0)).sum(
        spatial_dims, skipna=False
    )
    return xr.Dataset(out)


def spatial_std(
    field: xr.DataArray,
    area: xr.DataArray,
    wet: xr.DataArray,
    spatial_dims: tuple[str, str] = SPATIAL_DIMS,
) -> xr.DataArray:
    """Area-weighted spatial standard deviation of `field` over `wet`, per time."""
    w = area.where(wet, 0.0)
    total = w.sum(spatial_dims)
    x = field.where(wet, 0.0)
    mean = (w * x).sum(spatial_dims) / total
    var = (w * x**2).sum(spatial_dims) / total - mean**2
    return cast(xr.DataArray, np.sqrt(var.clip(min=0.0)))


def anomaly_variance(
    field: xr.DataArray,
    clim: xr.DataArray,
    area: xr.DataArray,
    wet: xr.DataArray,
    spatial_dims: tuple[str, str] = SPATIAL_DIMS,
    time_dim: str = "time",
) -> xr.DataArray:
    """Area-weighted mean over cells of the temporal variance of `field - clim`.

    The deseasonalized variability a rollout keeps. Comparing a model's value
    with the truth's measures variance collapse (ratio well below 1) or excess
    noise (well above 1), which RMSE does not separate.
    """
    anom = field - climatology_forecast(clim, field[time_dim], time_dim)
    var_map = anom.var(time_dim)
    w = area.where(wet, 0.0)
    return (w * var_map.where(wet, 0.0)).sum(spatial_dims) / w.sum(spatial_dims)


def skill_score(se_model: np.ndarray | float, se_ref: np.ndarray | float):
    """``1 - se_model / se_ref``. Works on sums or means alike (the weights cancel)."""
    return 1.0 - np.asarray(se_model, dtype=float) / np.asarray(se_ref, dtype=float)


def year_blocks(sums: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """Sum per-timestamp rows into calendar-year blocks, keeping only `years`.

    `sums` is indexed by timestamp (anything `kernels.coerce_datetime_values`
    accepts); all its columns are additive sums.

    Raises:
        ValueError: if one of `years` has no rows.
    """
    index = kernels.coerce_datetime_values(np.asarray(sums.index))
    blocks = sums.groupby(index.year).sum(min_count=1)
    missing = [y for y in years if y not in blocks.index]
    if missing:
        raise ValueError(f"No timestamps in year(s) {missing}")
    return blocks.loc[years]


def bootstrap_mean_ci(
    values: np.ndarray, samples: int, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of independent blocks.

    Used with one value per calendar year, which respects the strong
    autocorrelation within a year. Returns NaNs with fewer than
    `kernels.MIN_BOOTSTRAP_BLOCKS` blocks, where a percentile interval
    overstates itself, or when any block is not finite.
    """
    v = np.asarray(values, dtype=float)
    if (
        v.size < kernels.MIN_BOOTSTRAP_BLOCKS
        or samples <= 0
        or not np.isfinite(v).all()
    ):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(samples, v.size), replace=True).mean(axis=1)
    tail = (1.0 - level) / 2.0 * 100.0
    lo, hi = np.percentile(means, [tail, 100.0 - tail])
    return float(lo), float(hi)
