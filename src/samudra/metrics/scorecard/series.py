# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""The one pass over the fields: a rollout and its truth reduced to small tables.

Everything the scorecard reports is computed from three tables made here, so
re-scoring, comparing against a champion, or redrawing a figure never re-reads
the multi-gigabyte fields:

- `sums`: per channel and timestamp, the additive sums of `skill.time_sums`
  (squared errors of the model and of each reference, the signed error, the
  weight), plus the area-weighted spatial standard deviation of model and truth.
- `anomaly_variance`: per channel, the deseasonalized temporal variance of the
  model and of the truth over the whole rollout.
- `health`: per timestamp, quantities that should stay physical (extremes,
  volume means), for the model and the truth side by side.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import xarray as xr

from samudra.metrics import kernels, skill

logger = logging.getLogger(__name__)

SPATIAL_DIMS = skill.SPATIAL_DIMS
REFERENCES = ("persistence", "climatology")

# Values below these are not seawater: potential temperature below the freezing
# point of the saltiest water (about -2.2 degC at 40 psu, with a margin), and
# negative salinity. The share of ocean area beyond them is reported per step.
PHYSICAL_FLOORS = {"thetao": -2.5, "so": 0.0}

SUMS_FILE = "series_sums.parquet"
ANOMALY_FILE = "series_anomaly_variance.parquet"
HEALTH_FILE = "series_health.parquet"


@dataclass(frozen=True)
class Series:
    """The scorecard's inputs: small tables, written next to the scorecard."""

    sums: pd.DataFrame
    anomaly_variance: pd.DataFrame
    health: pd.DataFrame

    def write(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.sums.to_parquet(directory / SUMS_FILE, index=False)
        self.anomaly_variance.to_parquet(directory / ANOMALY_FILE, index=False)
        self.health.to_parquet(directory / HEALTH_FILE, index=False)

    @classmethod
    def read(cls, directory: Path) -> Series:
        return cls(
            pd.read_parquet(directory / SUMS_FILE),
            pd.read_parquet(directory / ANOMALY_FILE),
            pd.read_parquet(directory / HEALTH_FILE),
        )


def channel_parts(channel: str) -> tuple[str, int]:
    """``thetao_7`` -> (``thetao``, 7); a level-free channel like ``zos`` is level 0."""
    base, _, suffix = channel.rpartition("_")
    if base and suffix.isdigit():
        return base, int(suffix)
    return channel, 0


def cell_area(truth: xr.Dataset) -> xr.DataArray:
    """Cell area carried with the data (`areacello`), so no grid shape is assumed.

    Raises:
        ValueError: if the dataset carries no area. Rebuilding it from 1-D
            lat/lon would assume a rectilinear grid.
    """
    if "areacello" not in truth.variables:
        raise ValueError(
            "The truth dataset has no `areacello`. The scorecard weights by the "
            "area carried with the data rather than rebuilding it from lat/lon."
        )
    return _bare(truth["areacello"])


def _bare(field: xr.DataArray) -> xr.DataArray:
    """Drop spatial coordinates, keeping dims, time and pentad, so grids align by position."""
    drop = [c for c in field.coords if c not in ("time", "pentad")]
    return field.drop_vars(drop)


def _check_same_grid(predictions: xr.Dataset, truth: xr.Dataset) -> None:
    for dim in SPATIAL_DIMS:
        if predictions.sizes.get(dim) != truth.sizes.get(dim):
            raise ValueError(
                f"Rollout has {predictions.sizes.get(dim)} cells along {dim!r}, "
                f"the truth {truth.sizes.get(dim)}: score each rollout against "
                "the truth at its own resolution."
            )
        p, t = predictions.get(dim), truth.get(dim)
        if p is not None and t is not None and p.ndim == t.ndim == 1:
            if not np.allclose(p.values, t.values, atol=1e-6):
                raise ValueError(f"Rollout and truth {dim!r} coordinates differ")


def initial_state(truth: xr.Dataset, first_forecast_time) -> xr.Dataset:
    """The latest true state before the first forecast: the persistence forecast."""
    times = kernels.coerce_datetime_values(truth["time"].values)
    first = kernels.coerce_datetime_values(np.asarray([first_forecast_time]))[0]
    before = np.flatnonzero(times < first)
    if before.size == 0:
        raise ValueError(
            f"The truth has no state before the first forecast ({first}); it "
            "cannot provide the initial condition for persistence."
        )
    return truth.isel(time=int(before[-1]))


def compute(
    predictions: xr.Dataset,
    truth: xr.Dataset,
    climatology: xr.Dataset,
    channels: list[str],
) -> Series:
    """Reduce a rollout and its truth to the scorecard's tables.

    Args:
        predictions: The rollout, flattened channels (``thetao_0``...), dims
            (time, lat, lon).
        truth: The truth on the same grid, covering the rollout's timestamps
            and at least one earlier state (the initial condition).
        climatology: Pentad climatology per channel (`climatology.build`).
        channels: The prognostic channels to score.

    Raises:
        ValueError: on any mismatch of grid, timestamps or channels.
    """
    _check_same_grid(predictions, truth)
    missing = [c for c in channels if c not in predictions or c not in climatology]
    if missing:
        raise ValueError(f"Channels missing from the rollout or climatology: {missing}")
    times = predictions["time"]
    known = set(kernels.coerce_datetime_values(truth["time"].values))
    absent = [t for t in kernels.coerce_datetime_values(times.values) if t not in known]
    if absent:
        raise ValueError(
            f"{len(absent)} rollout timestamps are not in the truth (first: "
            f"{absent[0]}). Rollout and truth timestamps must match exactly; "
            "nothing is interpolated."
        )
    t0 = initial_state(truth, times.values[0])
    truth = truth.sel(time=times.values)
    area = cell_area(truth).load()

    sums: list[pd.DataFrame] = []
    anomaly: list[dict] = []
    for c in channels:
        # One channel is loaded whole and reduced in memory: thousands of tiny
        # per-step dask tasks cost far more than the arithmetic. A 1/4-degree
        # channel over ~600 steps is ~2.5 GB per field.
        f = _bare(predictions[c]).load()
        t = _bare(truth[c]).load()
        clim = _bare(climatology[c]).load()
        floor = PHYSICAL_FLOORS.get(channel_parts(c)[0], -np.inf)
        ds = _reduce(f, t, clim, _bare(t0[c]).load(), area, floor)
        frame = ds.drop_vars(["anomvar_model", "anomvar_truth"]).to_dataframe()
        sums.append(frame.reset_index().assign(channel=c))
        anomaly.append(
            {
                "channel": c,
                "model": float(ds["anomvar_model"]),
                "truth": float(ds["anomvar_truth"]),
            }
        )
        logger.info("scorecard: reduced %s", c)

    sums_frame = pd.concat(sums, ignore_index=True)
    sums_frame["time"] = kernels.coerce_datetime_values(sums_frame["time"].to_numpy())
    health = _health(sums_frame)
    return Series(sums_frame, pd.DataFrame(anomaly), health)


def _reduce(
    f: xr.DataArray,
    t: xr.DataArray,
    clim: xr.DataArray,
    t0: xr.DataArray,
    area: xr.DataArray,
    floor: float,
) -> xr.Dataset:
    """All of one channel's per-timestamp reductions (see the module docstring)."""
    times = t["time"]
    refs = {
        "persistence": t0.expand_dims(time=times.values),
        "climatology": skill.climatology_forecast(clim, times),
    }
    wet = cast(xr.DataArray, np.isfinite(t.isel(time=0, drop=True)))
    ds = skill.time_sums(f, t, refs, area)
    ds["std_model"] = skill.spatial_std(f, area, wet)
    ds["std_truth"] = skill.spatial_std(t, area, wet)
    ds["max_model"] = f.where(wet).max(SPATIAL_DIMS)
    ds["max_truth"] = t.where(wet).max(SPATIAL_DIMS)
    ds["min_model"] = f.where(wet).min(SPATIAL_DIMS)
    ds["min_truth"] = t.where(wet).min(SPATIAL_DIMS)
    w = area.where(wet, 0.0)
    ds["unphysical_model"] = w.where(f < floor, 0.0).sum(SPATIAL_DIMS)
    ds["unphysical_truth"] = w.where(t < floor, 0.0).sum(SPATIAL_DIMS)
    ds["anomvar_model"] = skill.anomaly_variance(f, clim, area, wet)
    ds["anomvar_truth"] = skill.anomaly_variance(t, clim, area, wet)
    return ds


def _health(sums: pd.DataFrame) -> pd.DataFrame:
    """Per-timestamp extremes, unphysical and non-finite area per variable.

    Area shares are of the wet cell area summed over the variable's levels.

    Drift of volume means needs no extra pass: it is the thickness-weighted sum
    of the channels' signed errors over their weights (see `card`).
    """
    rows = []
    by_var = sums.assign(var=sums["channel"].map(lambda c: channel_parts(c)[0]))
    for var, group in by_var.groupby("var"):
        per_time = group.groupby("time")
        rows.append(
            pd.DataFrame(
                {
                    "time": per_time.size().index,
                    "quantity": f"{var}_max",
                    "model": per_time["max_model"].max().to_numpy(),
                    "truth": per_time["max_truth"].max().to_numpy(),
                }
            )
        )
        rows.append(
            pd.DataFrame(
                {
                    "time": per_time.size().index,
                    "quantity": f"{var}_min",
                    "model": per_time["min_model"].min().to_numpy(),
                    "truth": per_time["min_truth"].min().to_numpy(),
                }
            )
        )
        rows.append(
            pd.DataFrame(
                {
                    "time": per_time.size().index,
                    "quantity": f"{var}_unphysical_area_fraction",
                    "model": (
                        per_time["unphysical_model"].sum() / per_time["weight"].sum()
                    ).to_numpy(),
                    "truth": (
                        per_time["unphysical_truth"].sum() / per_time["weight"].sum()
                    ).to_numpy(),
                }
            )
        )
        rows.append(
            pd.DataFrame(
                {
                    "time": per_time.size().index,
                    "quantity": f"{var}_nonfinite_area_fraction",
                    "model": (
                        per_time["nonfinite"].sum() / per_time["weight"].sum()
                    ).to_numpy(),
                    "truth": 0.0,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)
