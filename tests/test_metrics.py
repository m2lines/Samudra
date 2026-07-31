# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the observation-based metric kernels and driver.

Each test pins a distinct failure mode rather than a distinct function: an
area-weighting error, a depth-boundary error, a seasonal/trend removal error, a
silent time-misalignment, and a break in the end-to-end contract.
"""

import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from samudra.constants import build_om4_spec
from samudra.metrics import kernels, observations, report

OM4_SPEC = build_om4_spec()


def _grid(nlat: int = 9, nlon: int = 12) -> tuple[np.ndarray, np.ndarray]:
    return np.linspace(-70.0, 70.0, nlat), np.arange(15.0, 360.0, 360.0 / nlon)


def _area(lat: np.ndarray, lon: np.ndarray) -> xr.DataArray:
    return xr.DataArray(
        observations.spherical_cell_area(lat, lon),
        dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon},
    )


def test_area_weighting_follows_cell_area_not_cell_count():
    """Area weighting must track real cell area, and correlation must saturate at +-1.

    Catches the classic latitude bug where a naive mean treats a polar cell as
    equal to an equatorial one.
    """
    # An even latitude count, so the grid straddles the equator symmetrically
    # and no row sits exactly on it.
    lat, lon = _grid(8, 12)
    area = _area(lat, lon)

    # A field that is 1 in the northern hemisphere and -1 in the southern. The
    # area-weighted mean is 0 by symmetry, but the RMS is exactly 1.
    hemisphere = xr.DataArray(
        np.where(lat[:, None] > 0, 1.0, -1.0) * np.ones((lat.size, lon.size)),
        dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon},
    )
    assert kernels.area_weighted_mean(hemisphere, area) == pytest.approx(0.0, abs=1e-12)
    assert kernels.area_weighted_map_rmse(hemisphere, area) == pytest.approx(1.0)

    # Weighting by a non-uniform area must differ from the unweighted mean.
    banded = xr.DataArray(
        np.cos(np.deg2rad(lat))[:, None] * np.ones((1, lon.size)),
        dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon},
    )
    assert kernels.area_weighted_mean(banded, area) != pytest.approx(
        float(banded.mean()), abs=1e-6
    )

    rng = np.random.default_rng(0)
    field = xr.DataArray(
        rng.normal(size=(lat.size, lon.size)),
        dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon},
    )
    assert kernels.area_weighted_pattern_corr(field, field, area) == pytest.approx(1.0)
    assert kernels.area_weighted_pattern_corr(field, -field, area) == pytest.approx(
        -1.0
    )


def test_ohc_layers_partition_the_column_at_the_depth_boundary():
    """Depth layers must prorate the cell straddling 700 m, not include or drop it.

    Regression test: selecting layers by center depth made a nominal "0-700 m"
    integral actually cover 0-650 m on the OM4 grid, where the cell centered at
    775 m spans 650-900 m.
    """
    centers = np.array(OM4_SPEC.depth_levels)
    thickness = np.array(OM4_SPEC.depth_thickness)
    depth = xr.DataArray(centers, dims=["lev"], coords={"lev": centers})
    dz = xr.DataArray(thickness, dims=["lev"], coords={"lev": centers})

    upper = kernels.layer_overlap_thickness(depth, 0, 700, native_dz=dz)
    lower = kernels.layer_overlap_thickness(depth, 700, 2000, native_dz=dz)

    # Each layer integrates exactly its own nominal thickness.
    assert float(upper.sum()) == pytest.approx(700.0)
    assert float(lower.sum()) == pytest.approx(1300.0)

    # The straddling cell (650-900 m) is split, not assigned wholesale.
    straddling = 775.0
    assert float(upper.sel(lev=straddling)) == pytest.approx(50.0)
    assert float(lower.sel(lev=straddling)) == pytest.approx(200.0)

    # Together the layers tile [0, 2000) without double counting.
    assert float((upper + lower).sum()) == pytest.approx(2000.0)


def test_residual_variance_removes_an_exact_seasonal_cycle():
    """A purely seasonal signal must leave no residual variance.

    Boundary case chosen so the assertion is exact: the signal is constant
    within each pentad bin, so the pentad climatology removes all of it.
    """
    time = pd.date_range("2015-01-01", "2022-12-31", freq="5D")
    season = 3.0 * np.sin(2 * np.pi * kernels.pentad_index(time) / 73.0)

    series = pd.Series(season, index=time)
    assert kernels.series_residual_variance(series) == pytest.approx(0.0, abs=1e-20)

    field = xr.DataArray(
        season[:, None, None] * np.ones((1, 2, 2)),
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    )
    assert float(kernels.residual_variance_map(field).max()) == pytest.approx(
        0.0, abs=1e-20
    )

    # A pure linear trend is recovered exactly, in units per year.
    trend = pd.Series(0.002 * np.arange(len(time)), index=time)
    assert kernels.series_linear_trend_per_year(trend) == pytest.approx(
        0.002 / 5 * 365.25
    )


def test_misaligned_timestamps_raise_instead_of_interpolating():
    """Offset time axes must fail loudly rather than be silently interpolated.

    Temporal interpolation deflates model variance and inflates RMSE, so a
    mismatch has to surface as an error the caller resolves.
    """
    values = [1.0, 2.0, 3.0]
    model = xr.DataArray(
        values,
        dims=["time"],
        coords={"time": pd.to_datetime(["2020-01-01", "2020-01-06", "2020-01-11"])},
    )
    offset = xr.DataArray(
        values,
        dims=["time"],
        coords={"time": pd.to_datetime(["2020-01-02", "2020-01-07", "2020-01-12"])},
    )

    with pytest.raises(ValueError, match="must match exactly"):
        kernels.require_exact_time_match(model, offset, "sst")

    # Identical axes pass without complaint.
    kernels.require_exact_time_match(model, model.copy(), "sst")

    # Products legitimately end on different dates, so the driver trims to the
    # shared span rather than demanding identical coverage...
    stamps = pd.to_datetime(["2020-01-01", "2020-01-06", "2020-01-11", "2020-01-16"])
    long_model = xr.DataArray(
        [1.0, 2.0, 3.0, 4.0], dims=["time"], coords={"time": stamps}
    )
    short_obs = long_model.isel(time=slice(0, 3))
    trimmed_model, trimmed_obs = report._aligned(
        long_model,
        short_obs,
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01")),
        "sst",
    )
    assert trimmed_model.sizes["time"] == trimmed_obs.sizes["time"] == 3

    # ...but a gap *inside* the shared span is still an error, since that is a
    # genuine misalignment rather than a coverage difference.
    gapped_obs = long_model.isel(time=[0, 2, 3])
    with pytest.raises(ValueError, match="must match exactly"):
        report._aligned(
            long_model,
            gapped_obs,
            (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01")),
            "sst",
        )


def test_model_grid_adapter_preserves_geometry_and_refuses_curvilinear():
    """The rollout layout must convert to lat/lon without losing real geometry.

    Rollouts carry 2-D lat/lon coordinates alongside (y, x) dims, so a naive
    rename collides; and on a curvilinear grid y/x are not degrees at all.
    """
    lat, lon = _grid(6, 6)
    rollout = xr.Dataset(
        {"zos": (("time", "y", "x"), np.zeros((2, 6, 6)))},
        coords={
            "y": lat,
            "x": lon,
            "lat": (("y", "x"), np.broadcast_to(lat[:, None], (6, 6)).copy()),
            "lon": (("y", "x"), np.broadcast_to(lon[None, :], (6, 6)).copy()),
            "areacello": (("y", "x"), np.full((6, 6), 7.0)),
            "time": pd.to_datetime(["2015-01-01", "2015-01-06"]),
        },
    )

    adapted = observations.model_on_latlon_grid(rollout, OM4_SPEC)
    assert set(adapted["zos"].dims) == {"time", "lat", "lon"}
    # The model's own cell area survives, rather than being re-derived.
    assert float(observations.model_cell_area(adapted).mean()) == pytest.approx(7.0)

    with pytest.raises(NotImplementedError, match="rectilinear"):
        observations.model_on_latlon_grid(rollout, build_om4_spec(grid_type="tripolar"))


def test_cftime_rollouts_become_comparable():
    """A cftime time axis must be normalised, or window selection cannot happen.

    Regression test: OM4 stores time as `cftime.DatetimeJulian`, and every
    rollout inherits it. Selecting a window with pandas timestamps then raises
    "cannot compare ... different calendars", which broke real eval runs while
    synthetic datetime64 fixtures passed.
    """
    lat, lon = _grid(6, 6)
    times = [cftime.DatetimeJulian(2021, 1, 1 + 5 * i, 12) for i in range(4)]
    rollout = xr.Dataset(
        {"zos": (("time", "y", "x"), np.zeros((4, 6, 6)))},
        coords={
            "y": lat,
            "x": lon,
            "lat": (("y", "x"), np.broadcast_to(lat[:, None], (6, 6)).copy()),
            "lon": (("y", "x"), np.broadcast_to(lon[None, :], (6, 6)).copy()),
            "time": times,
        },
    )
    assert not np.issubdtype(rollout["time"].dtype, np.datetime64)

    adapted = observations.model_on_latlon_grid(rollout, OM4_SPEC)
    assert np.issubdtype(adapted["time"].dtype, np.datetime64)
    # The operation that actually failed in production: a pandas-timestamp
    # window. Samples sit at 12:00, so the end bound is taken past that hour to
    # include the third of the four timestamps.
    selected = adapted.sel(
        time=slice(pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-12"))
    )
    assert selected.sizes["time"] == 3


def _synthetic_case(seed: int = 0):
    """A tiny but complete model/observation set spanning two calendar years."""
    rng = np.random.default_rng(seed)
    time = pd.date_range("2021-01-01", "2022-12-31", freq="5D")
    months = pd.date_range("2021-01-01", "2022-12-01", freq="MS")
    mlat, mlon = _grid(9, 12)
    olat, olon = _grid(13, 18)
    nlev = 12
    levs = np.array(OM4_SPEC.depth_levels[:nlev])

    rollout = xr.Dataset(
        {
            "thetao": (
                ("time", "lev", "y", "x"),
                10 + rng.normal(size=(len(time), nlev, mlat.size, mlon.size)),
            ),
            "zos": (
                ("time", "y", "x"),
                rng.normal(scale=0.1, size=(len(time), mlat.size, mlon.size)),
            ),
        },
        coords={
            "y": mlat,
            "x": mlon,
            "lat": (
                ("y", "x"),
                np.broadcast_to(mlat[:, None], (mlat.size, mlon.size)).copy(),
            ),
            "lon": (
                ("y", "x"),
                np.broadcast_to(mlon[None, :], (mlat.size, mlon.size)).copy(),
            ),
            "lev": levs,
            "areacello": (("y", "x"), np.full((mlat.size, mlon.size), 1e10)),
            "time": time,
        },
    )

    def product(data_vars, times, extra_coords=None):
        coords = {"lat": olat, "lon": olon, "time": times}
        coords.update(extra_coords or {})
        return observations.with_cell_area(
            observations.standardize(xr.Dataset(data_vars, coords=coords))
        )

    shape = (len(time), olat.size, olon.size)
    duacs = product(
        {
            "ugos": (("time", "lat", "lon"), rng.normal(scale=0.2, size=shape)),
            "vgos": (("time", "lat", "lon"), rng.normal(scale=0.2, size=shape)),
        },
        time,
    )
    oisst = product(
        {"sst": (("time", "lat", "lon"), 10 + rng.normal(size=shape))}, time
    )

    adepth = np.array([5.0, 50.0, 150.0, 400.0, 800.0, 1400.0, 1900.0])
    argo = product(
        {
            "temp": (
                ("time", "depth", "lat", "lon"),
                10 + rng.normal(size=(len(months), adepth.size, olat.size, olon.size)),
            )
        },
        months,
        {"depth": adepth},
    )
    return rollout, duacs, oisst, argo


def test_driver_reports_every_headline_metric_as_a_plain_float():
    """End-to-end: the driver emits the full headline set, with a namespaced baseline.

    Guards the contract the W&B report depends on -- key names, plain float
    values, and per-year detail behind each uncertainty interval.
    """
    rollout, duacs, oisst, argo = _synthetic_case()
    model = observations.model_on_latlon_grid(rollout, OM4_SPEC)
    dz = observations.model_depth_thickness(model, OM4_SPEC)

    frame = report.compute_observation_metrics(
        {"model": model, "om4": model},
        duacs=duacs,
        oisst=oisst,
        argo=argo,
        model_dz={"model": dz, "om4": dz},
        window=(pd.Timestamp("2021-01-01"), pd.Timestamp("2022-12-31")),
        bootstrap_samples=64,
    )

    assert list(frame.columns) == report.COLUMNS
    # Per-year rows back the uncertainty: two complete calendar years each, for
    # the five RMSE metrics, for both models.
    assert (frame["period_kind"] == "annual").sum() == 2 * 5 * 2

    scalars = report.to_wandb(frame, "model")
    expected = {
        "obs/velocity/total_rmse",
        "obs/eke/total_rmse",
        "obs/sst/total_rmse",
        "obs/ohc_0_700/per_area_total_rmse",
        "obs/ohc_700_2000/per_area_total_rmse",
        "obs/sst/residual_variance_map_rmse",
        "obs/sst/residual_variance_pattern_corr",
        "obs/ohc_upper700/residual_variance_map_rmse",
        "obs/ohc_upper700/residual_variance_pattern_corr",
    }
    assert expected <= set(scalars)
    # The baseline is namespaced so it cannot shadow the model under evaluation.
    assert {key.replace("obs/", "obs/om4/") for key in expected} <= set(scalars)
    assert all(isinstance(value, float) for value in scalars.values())

    # Every RMSE metric carries a bootstrap interval that brackets its estimate.
    # MetricsDict admits W&B media types too, so narrow to float for comparison.
    for key in expected:
        if not key.endswith("total_rmse"):
            continue
        low = float(scalars[f"{key}_ci_low"])  # type: ignore[arg-type]
        high = float(scalars[f"{key}_ci_high"])  # type: ignore[arg-type]
        assert low <= float(scalars[key]) <= high  # type: ignore[arg-type]


def test_incomplete_calendar_years_are_rejected():
    """A ragged final year must fail rather than silently skew the equal-year blocks."""
    rollout, duacs, oisst, argo = _synthetic_case()
    model = observations.model_on_latlon_grid(rollout, OM4_SPEC)

    with pytest.raises(ValueError, match="complete calendar years"):
        report.compute_observation_metrics(
            {"model": model},
            duacs=duacs,
            oisst=oisst,
            argo=argo,
            model_dz={"model": observations.model_depth_thickness(model, OM4_SPEC)},
            # Stops mid-year, so 2022 is only partially covered.
            window=(pd.Timestamp("2021-01-01"), pd.Timestamp("2022-06-30")),
            bootstrap_samples=0,
        )
