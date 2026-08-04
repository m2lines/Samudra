# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the observation-based metric kernels and driver.

Each test pins a distinct failure mode rather than a distinct function: an
area-weighting error, a depth-boundary error, a seasonal/trend removal error, a
silent time-misalignment, and a break in the end-to-end contract.
"""

import logging
from typing import cast

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


def test_ohc_drops_columns_that_run_out_before_the_layer_does():
    """A column shallower than the layer must not book its missing water as error.

    `sum(skipna=True)` treats a masked level as zero heat, so a cell where the
    model and observation bathymetry disagree by one level would contribute a
    deficit the size of the metric itself -- about 2e9 J m^-2 for 50 m. The
    axis-level guard cannot see this: it fires when the whole grid is too
    shallow, while this is a single column running out early.
    """
    centers = np.array(OM4_SPEC.depth_levels)
    dz = xr.DataArray(
        np.array(OM4_SPEC.depth_thickness), dims=["lev"], coords={"lev": centers}
    )
    field = xr.DataArray(
        np.full((centers.size, 1, 2), 10.0),
        dims=["lev", "lat", "lon"],
        coords={"lev": centers, "lat": [0.0], "lon": [0.0, 1.0]},
    )
    # The first column stops at 650 m; the second spans the full grid.
    field = field.where((field["lev"] < 650) | (field["lon"] > 0.5))

    ohc = kernels.ohc_per_area_layer_maps(field, native_dz=dz, depth_name="lev")
    surface = ohc[kernels.OHC_LAYERS[0].label]
    assert np.isnan(float(surface.values[0, 0]))
    assert float(surface.values[0, 1]) > 0


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

    # The OM4 baseline does not arrive in the rollout layout: a DataSource has
    # already run it through `with_lat_lon_coords`, which renames y/x to 1-D
    # lat/lon and parks the 2-D geography on lat_2d/lon_2d. Rejecting that shape
    # crashed every eval with observations enabled, at the end of the rollout.
    from samudra.utils.data import with_lat_lon_coords

    standardized = with_lat_lon_coords(rollout)
    assert set(standardized.dims) == {"time", "lat", "lon"}
    from_source = observations.model_on_latlon_grid(standardized, OM4_SPEC)
    assert set(from_source["zos"].dims) == {"time", "lat", "lon"}
    assert float(observations.model_cell_area(from_source).mean()) == pytest.approx(7.0)
    # Both entry paths must land on the same grid, or model and baseline would
    # be scored against different geometry.
    assert from_source["lat"].equals(adapted["lat"])
    assert from_source["lon"].equals(adapted["lon"])


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


def test_partial_edge_months_and_seam_are_handled():
    """Two silent-corruption paths that only show up on real-shaped inputs.

    A month at the edge of a record is partial, but reduces to the same single
    monthly label as a full month on the other side. And a global longitude
    derivative taken without wrapping differences the seam meridians against
    the wrong neighbour.
    """
    # A rollout that starts mid-October and stops on 24 December: both edge
    # months are short and must not be compared against full months.
    time = pd.date_range("2014-10-20 12:00", "2022-12-24 12:00", freq="5D")
    series = xr.DataArray(np.ones(len(time)), dims=["time"], coords={"time": time})
    kept = pd.DatetimeIndex(
        kernels.monthly_mean_of_complete_months(series)["time"].values
    )
    assert kept[0].strftime("%Y-%m") == "2014-11"
    assert kept[-1].strftime("%Y-%m") == "2022-11"

    # A monthly product must survive intact: one sample *is* the whole month.
    months = pd.date_range("2015-01-01", "2022-12-01", freq="MS")
    monthly = xr.DataArray(np.ones(len(months)), dims=["time"], coords={"time": months})
    assert kernels.monthly_mean_of_complete_months(monthly).sizes["time"] == len(months)

    # A global model grid's outermost centers sit inside the observation
    # product's, so plain interpolation would call those observation columns
    # out of bounds and silently drop them from every reduction.
    model_lon = np.arange(0.125, 360.0, 0.25)
    obs_lon = np.arange(0.0625, 360.0, 0.125)
    rows = np.array([-1.0, 1.0])
    model = xr.DataArray(
        np.ones((2, model_lon.size)),
        dims=["lat", "lon"],
        coords={"lat": rows, "lon": model_lon},
    )
    obs = xr.DataArray(
        np.ones((2, obs_lon.size)),
        dims=["lat", "lon"],
        coords={"lat": rows, "lon": obs_lon},
    )
    assert not np.isnan(kernels.model_field_on_obs_grid(model, obs).values).any()
    chunked_model = model.chunk({"lon": 360})
    assert not np.isnan(
        kernels.model_field_on_obs_grid(chunked_model, obs).compute().values
    ).any()
    # A regional grid has real edges, so it must pass through untouched.
    regional = model.isel(lon=slice(0, 40))
    assert kernels._wrap_lon(regional, "lon").equals(regional)

    # On a global grid the seam is an interior point, so a periodic field
    # differentiates exactly there.
    lon = np.arange(0.0, 360.0, 10.0)
    field = xr.DataArray(
        np.cos(np.deg2rad(lon))[None, :],
        dims=["y", "x"],
        coords={"y": [30.0], "x": lon},
    )
    analytic = -np.sin(np.deg2rad(lon)) * np.pi / 180
    wrapped = kernels._differentiate_lon(field, "x").values[0]
    assert wrapped[0] == pytest.approx(analytic[0], abs=1e-12)

    # Dask-backed too. Padding leaves 1-wide chunks at the edges and
    # `np.gradient` requires two points per chunk, so a numpy-only test passes
    # while every real rollout -- which is always chunked -- raises.
    chunked = kernels._differentiate_lon(field.chunk({"x": 12}), "x").compute()
    assert chunked.values[0][0] == pytest.approx(analytic[0], abs=1e-12)
    # A regional grid genuinely has edges, and must be left alone.
    regional = field.isel(x=slice(0, 8))
    assert np.allclose(
        kernels._differentiate_lon(regional, "x").values,
        regional.differentiate("x").values,
    )


def test_a_year_without_coverage_is_refused_not_dropped():
    """An empty year must fail, not silently shrink the equal-year block set."""
    values = np.array([0.5, np.inf, 0.6, 0.7])
    summary = kernels.interannual_rmse_summary(values, bootstrap_samples=0)
    # The summary itself still filters -- which is exactly why the caller has to
    # catch the empty year before it reaches here.
    assert summary["n_years"] == 3

    lat, lon = _grid(8, 12)
    area = _area(lat, lon)
    time = pd.date_range("2021-01-01", "2022-12-31", freq="5D")
    err = xr.DataArray(
        np.ones((len(time), lat.size, lon.size)),
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": lat, "lon": lon},
    )
    # Blank 2022 entirely: no finite paired cell that year.
    err = err.where(err["time"].dt.year != 2022)
    # Subsumed by the stronger coverage check: a year with no finite cell is
    # simply the 0% case of "too little paired data".
    with pytest.raises(ValueError, match="too little paired data"):
        kernels.rmse_map_with_uncertainty(
            err, area, ("lat", "lon"), "ctx", bootstrap_samples=0
        )


def _synthetic_case(seed: int = 0):
    """A tiny but complete model/observation set spanning two calendar years."""
    rng = np.random.default_rng(seed)
    time = pd.date_range("2021-01-01", "2022-12-31", freq="5D")
    months = pd.date_range("2021-01-01", "2022-12-01", freq="MS")
    mlat, mlon = _grid(9, 12)
    olat, olon = _grid(13, 18)
    # The full depth axis, not a truncated one: the OHC layers integrate to
    # 2000 m, and a shallower column would silently compare a partial integral
    # against the observation product's full one.
    nlev = len(OM4_SPEC.depth_levels)
    levs = np.array(OM4_SPEC.depth_levels)

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


def test_ohc_scores_only_whole_calendar_years():
    """A year missing a month must be dropped, not scored against full years.

    Dropping partial edge months means a rollout ending 24 December contributes
    no December at all, so its final year is genuinely incomplete. Scoring it
    would put an 11-month year in the same equal-weight block set as 12-month
    ones -- and the year guard downstream can only report that *some* year is
    ragged, not which month is missing.
    """
    months = pd.date_range("2021-01-01", "2022-12-01", freq="MS")
    full = xr.DataArray(np.ones(len(months)), dims=["time"], coords={"time": months})
    kept, _ = report._whole_years(full, full, "full coverage")
    assert kept.sizes["time"] == 24

    # Same series without its final December: 2022 goes, 2021 survives intact.
    short = full.isel(time=slice(0, -1))
    trimmed, _ = report._whole_years(short, short, "missing december")
    labels = pd.DatetimeIndex(trimmed["time"].values)
    assert trimmed.sizes["time"] == 12
    assert labels[0].year == labels[-1].year == 2021

    # No complete year at all is an error, not an empty result that would
    # surface later as an unexplained division by zero.
    partial = pd.date_range("2021-06-01", "2021-09-01", freq="MS")
    stub = xr.DataArray(np.ones(len(partial)), dims=["time"], coords={"time": partial})
    with pytest.raises(ValueError, match="twelve whole months"):
        report._whole_years(stub, stub, "no complete year")


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

    # This case spans two calendar years, which is below the block count a
    # percentile bootstrap needs to say anything -- so no interval is offered,
    # and the frame records why rather than labelling a two-point spread a 95%
    # CI. Where an interval *is* present it must bracket its own estimate.
    primary = frame[frame["period_kind"] == "primary_complete_years"]
    assert (primary["uncertainty_method"] == "insufficient_blocks").all()
    assert primary["ci_low"].isna().all()
    for key in expected:
        if f"{key}_ci_low" not in scalars:
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


def test_whole_years_drops_the_years_it_reports_dropping(caplog):
    """A ragged year in the middle of the record must go, not just be logged.

    `_whole_years` promises that "a year missing a month has to go rather than
    be silently down-weighted", but it only narrows the outer bound, so an
    interior ragged year survives underneath a log line announcing its removal.
    """
    months = pd.DatetimeIndex(
        list(pd.date_range("2021-01-01", "2021-12-01", freq="MS"))
        # 2022 is missing May and June.
        + [
            stamp
            for stamp in pd.date_range("2022-01-01", "2022-12-01", freq="MS")
            if stamp.month not in (5, 6)
        ]
        + list(pd.date_range("2023-01-01", "2023-12-01", freq="MS"))
    )
    series = xr.DataArray(np.ones(len(months)), dims=["time"], coords={"time": months})

    with caplog.at_level(logging.INFO, logger="samudra.metrics.report"):
        kept, _ = report._whole_years(series, series, "interior gap")

    assert "dropped partly covered year(s) [2022]" in caplog.text
    assert set(pd.DatetimeIndex(kept["time"].values).year) == {2021, 2023}


def test_an_equal_year_block_needs_data_not_only_timestamps():
    """A year has to carry data to earn its block, not just keep its stamps.

    The empty-year guard fires only when a year has *no* finite paired cell.
    A year that keeps all 73 timestamps but carries data at four of them still
    counts as one full-weight calendar block, which is exactly the silent
    down-weighting `require_complete_calendar_years` exists to prevent. The
    products make this reachable: one missing DUACS day NaNs five consecutive
    5-day means everywhere, via `min_periods=5` in the rolling alignment.
    """
    lat, lon = _grid(8, 12)
    time = pd.date_range("2021-01-03 12:00", "2022-12-29 12:00", freq="5D")
    error_squared = xr.DataArray(
        np.ones((len(time), lat.size, lon.size)),
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": lat, "lon": lon},
    )
    in_2022 = error_squared["time"].dt.year == 2022
    first_four = xr.DataArray(
        np.isin(np.arange(len(time)), np.where(in_2022.values)[0][:4]),
        dims=["time"],
        coords={"time": time},
    )
    # 2022 keeps every timestamp; only four of them carry a (badly wrong) value.
    sparse = xr.where(in_2022 & first_four, 100.0, error_squared.where(~in_2022))

    with pytest.raises(ValueError, match="too little paired data") as raised:
        kernels.rmse_map_with_uncertainty(
            sparse, _area(lat, lon), ("lat", "lon"), "sparse year", bootstrap_samples=0
        )
    # The message has to name the year and its coverage, or an operator hitting
    # this after a multi-hour rollout cannot tell which product is at fault.
    assert "2022" in str(raised.value)


def test_a_calendar_year_missing_a_fortnight_is_not_complete():
    """ "Complete" has to mean complete, or equal-year blocks are not equal.

    `min_sample_fraction=0.95` on a 73-sample year tolerates three missing
    steps, and the `1.5 * step` edge tolerance allows about 7.5 more days at
    each end -- so a "complete" year can be missing a month of samples and
    still be weighted identically to a full one.
    """
    time = pd.date_range("2021-01-03 12:00", "2022-12-29 12:00", freq="5D")
    hole_start = np.where(time.year == 2022)[0][36]
    keep = np.ones(len(time), dtype=bool)
    keep[hole_start : hole_start + 3] = False  # a 15-day hole mid-2022
    ragged = time[keep]

    years = kernels.complete_calendar_years(
        xr.DataArray(ragged, dims=["time"], coords={"time": ragged})
    )
    assert years == [2021]


def test_an_undersampled_cell_has_no_residual_variance_rather_than_zero():
    """Too few samples must give NaN, not a perfectly quiescent ocean point.

    Deseasonalising subtracts a per-bin climatology, so a cell whose valid
    samples fall one per bin is compared against itself and residualises to
    exactly 0.0 -- indistinguishable from real quiescence. It then flows into
    the variance-map RMSE and the pattern correlation as model error. Twelve
    valid monthly samples is enough to trigger it, so this is not a degenerate
    one-or-two-sample corner.
    """
    months = pd.date_range("2015-01-01", "2022-12-01", freq="MS")
    rng = np.random.default_rng(1)
    field = xr.DataArray(
        rng.normal(scale=3.0, size=(len(months), 2, 2)),
        dims=["time", "lat", "lon"],
        coords={"time": months, "lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    )
    seen: set[int] = set()
    one_per_calendar_month = []
    for stamp in months:
        one_per_calendar_month.append(stamp.month not in seen)
        seen.add(stamp.month)
    valid = np.ones((len(months), 2, 2), dtype=bool)
    valid[:, 0, 0] = one_per_calendar_month

    variance = kernels.residual_variance_map(
        field.where(xr.DataArray(valid, dims=field.dims, coords=field.coords))
    )
    assert float(variance[1, 1]) > 0.0  # the well-sampled neighbour
    assert np.isnan(float(variance[0, 0]))


def test_uncertainty_is_withheld_when_there_are_too_few_blocks():
    """Two calendar years cannot support a 95% interval, so none should be given.

    With two blocks the percentile bootstrap degenerates to [min, max] of the
    two annual RMSEs, which is still reported as a 10,000-draw 95% CI.
    """
    summary = kernels.interannual_rmse_summary(np.array([1.0, 2.0]))

    assert summary["n_years"] == 2
    assert np.isnan(summary["ci_low"])
    assert np.isnan(summary["ci_high"])


def test_the_scored_window_is_reported_per_metric():
    """OHC and SST can score different spans, so one `n_years` scalar cannot serve.

    Dropping partial months costs OHC its final year while the 5-day metrics
    keep theirs, and `to_wandb` collapses that to `max()` -- so W&B advertises
    the span of whichever metric scored the most years.
    """
    frame = pd.DataFrame(
        [
            {
                "metric": "surface_sst_total_rmse",
                "model": "m",
                "depth": None,
                "value": 0.8,
                "period_kind": "primary_complete_years",
                "n_years": 8.0,
            },
            {
                "metric": "ohc_per_area_total_rmse",
                "model": "m",
                "depth": kernels.OHC_LAYERS[0].label,
                "value": 3.0e9,
                "period_kind": "primary_complete_years",
                "n_years": 7.0,
            },
        ]
    )

    scalars = report.to_wandb(frame, "m")
    per_metric = {
        key: value for key, value in scalars.items() if key.endswith("_n_years")
    }
    assert per_metric == {
        "obs/sst/total_rmse_n_years": 8.0,
        "obs/ohc_0_700/per_area_total_rmse_n_years": 7.0,
    }, f"got {per_metric} alongside obs/n_years={scalars.get('obs/n_years')}"


def test_coastal_erosion_is_measured_and_reported():
    """Regridding erodes a coastal band, and how much must be visible.

    Linear interpolation NaNs an observation cell whenever its model-grid
    neighbours include land, so the comparison set loses a band one model cell
    wide -- all coastal, which is where model error is largest. The eroded
    fraction therefore *scales with model resolution*: a 1 degree run is scored
    on a smaller and easier subset of the ocean than a quarter degree one.

    That asymmetry is a known limitation, not something this test asserts away.
    What it pins is that the loss is measured and travels with the numbers, so
    a cross-resolution comparison can be checked rather than assumed. Removing
    the asymmetry means choosing a convention for model values at coastal
    points -- nearest-ocean extrapolation, or a fixed comparison mask -- which
    changes every published number and needs scientific sign-off.
    """
    obs_lat = np.arange(-89.875, 90.0, 0.25)
    obs_lon = np.arange(0.125, 360.0, 0.25)

    def is_land(lon, lat):
        return ((lon > 60) & (lon < 160) & (lat > -40) & (lat < 60)) | (
            np.abs(lat) > 85
        )

    lon_grid, lat_grid = np.meshgrid(obs_lon, obs_lat)
    obs = xr.DataArray(
        np.where(is_land(lon_grid, lat_grid), np.nan, 1.0),
        dims=["lat", "lon"],
        coords={"lat": obs_lat, "lon": obs_lon},
    )

    measured = {}
    for resolution in (1.0, 0.5, 0.25):
        model_lat = np.arange(-90.0 + resolution / 2, 90.0, resolution)
        model_lon = np.arange(resolution / 2, 360.0, resolution)
        mlon_grid, mlat_grid = np.meshgrid(model_lon, model_lat)
        model = xr.DataArray(
            np.where(is_land(mlon_grid, mlat_grid), np.nan, 1.0),
            dims=["lat", "lon"],
            coords={"lat": model_lat, "lon": model_lon},
        )
        paired = kernels.model_field_on_obs_grid(model, obs)
        recorded = report._pairing(paired, obs)

        # The reported fraction is the real one, not an estimate.
        ocean_cells = int(np.isfinite(obs).sum())
        assert recorded["n_paired_cells"] == float(int(np.isfinite(paired).sum()))
        assert recorded["paired_ocean_fraction"] == pytest.approx(
            recorded["n_paired_cells"] / ocean_cells
        )
        measured[resolution] = recorded["paired_ocean_fraction"]

    # A coarser model loses more of the ocean. Pinned so that any future change
    # to the pairing convention is a deliberate decision rather than a drift.
    assert measured[1.0] < measured[0.5] < measured[0.25]
    assert measured[1.0] == pytest.approx(0.989, abs=0.002)
    assert measured[0.25] == pytest.approx(0.997, abs=0.002)


def test_year_coverage_measures_missing_data_not_land():
    """The coverage guard must divide by the pairable ocean, not the whole globe.

    `rmse_map_with_uncertainty` documents `covered` as the "area-weighted
    fraction of the *paired* cells that actually carry data" and raises "too
    little paired data" below 50%. It divides by `xr.ones_like(error_squared)`
    instead -- every cell on the grid, land included -- so the number it tests
    is the finite fraction of the *planet*.

    The observation grids are global, and land is simply NaN on them: roughly
    29% of a lat/lon grid by area, plus the +-5 degree band
    `apply_equatorial_mask` blanks for DUACS, plus every shelf and marginal sea
    for the 700-2000 m OHC layer. A DUACS velocity year with perfect coverage
    therefore scores around 0.6 against a 0.50 threshold, and the deep OHC
    layer lower still. The guard is one masked marginal sea away from failing
    an eval, after the whole rollout has been paid for, because the Earth has
    continents.

    The opposite direction -- a year that keeps its timestamps but carries
    almost no data must still fail -- is already pinned by
    `test_an_equal_year_block_needs_data_not_only_timestamps`, so this asserts
    only the false positive.
    """
    lat, lon = _grid(9, 12)
    area = _area(lat, lon)
    time = pd.date_range("2021-01-01", "2022-12-31", freq="5D")
    error_squared = xr.DataArray(
        np.ones((len(time), lat.size, lon.size)),
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": lat, "lon": lon},
    )

    # Land is not missing data. Every ocean cell here carries a value at every
    # timestamp of both years; the only thing "missing" is the 58% of the grid
    # that is permanently dry.
    land = xr.DataArray(np.arange(lon.size) < 7, dims=["lon"], coords={"lon": lon})
    rmse_map, annual, _ = kernels.rmse_map_with_uncertainty(
        error_squared.where(~land), area, ("lat", "lon"), "land", bootstrap_samples=0
    )
    assert set(annual) == {2021, 2022}
    assert np.isfinite(rmse_map).any()


def test_regridding_keeps_the_polar_rows_it_keeps_the_seam_columns_for():
    """Latitude needs the treatment longitude got, or the loss is silent.

    `model_field_on_obs_grid` carries a paragraph explaining that a global model
    grid's outermost longitude centers sit inside the observation product's, so
    plain interpolation "calls those observation columns out of bounds and drops
    them from every reduction, so the metric quietly depends on grid alignment".
    The fix -- `_wrap_lon` -- applies to longitude only.

    Latitude has the same geometry and no wrap to exploit: a 1 degree model's
    outermost row sits at -89.5, OISST's at -89.875. Four quarter-degree
    observation rows fall outside the model's convex hull and are NaN'd out of
    every RMSE, variance map and pattern correlation -- silently, and by an
    amount that depends on the model's resolution, which is exactly the
    "quietly depends on grid alignment" failure the longitude comment rejects.
    Clamping the interpolation at the poles fixes it; nothing here does.

    `test_partial_edge_months_and_seam_are_handled` covers the longitude half of
    this on a two-row grid, where latitude cannot be out of bounds.
    """
    obs_lat = np.arange(-89.875, 90.0, 0.25)
    obs_lon = np.arange(0.125, 360.0, 0.25)
    obs = xr.DataArray(
        np.ones((obs_lat.size, obs_lon.size)),
        dims=["lat", "lon"],
        coords={"lat": obs_lat, "lon": obs_lon},
    )

    lost = {}
    for resolution in (1.0, 0.25):
        model_lat = np.arange(-90.0 + resolution / 2, 90.0, resolution)
        model_lon = np.arange(resolution / 2, 360.0, resolution)
        model = xr.DataArray(
            np.full((model_lat.size, model_lon.size), 3.0),
            dims=["lat", "lon"],
            coords={"lat": model_lat, "lon": model_lon},
        )
        paired = kernels.model_field_on_obs_grid(model, obs)
        # Rows of the observation grid that lost every single cell.
        missing = cast(xr.DataArray, ~np.isfinite(paired))
        lost[resolution] = int(missing.all("lon").sum())

    assert lost == {1.0: 0, 0.25: 0}, (
        f"whole observation latitude rows dropped, by model resolution: {lost}"
    )


def test_the_pairing_diagnostic_reaches_the_emitted_frame():
    """`_pairing` is dead code, so its two columns are always NaN.

    `report.COLUMNS` is called "the contract", and it declares `n_paired_cells`
    and `paired_ocean_fraction`. `_pairing` computes them, and its docstring
    explains why they matter: regridding erodes a coastal band whose width
    scales with model resolution, so "a coarse run is scored on a smaller and
    easier subset of the ocean than a fine one. Recording it makes that visible
    instead of leaving cross-resolution comparison merely 'not comparable' in
    prose."

    Nothing in the driver ever calls `_pairing`. Both columns are back-filled
    with NaN by `compute_observation_metrics`, so the CSV and the W&B table
    carry two empty columns and the erosion stays as invisible as before.
    `test_coastal_erosion_is_measured_and_reported` above calls the helper
    directly, so the suite stays green while the property that docstring claims
    -- that the loss travels with the numbers -- is false.
    """
    rollout, duacs, oisst, argo = _synthetic_case()
    model = observations.model_on_latlon_grid(rollout, OM4_SPEC)

    frame = report.compute_observation_metrics(
        {"model": model},
        duacs=duacs,
        oisst=oisst,
        argo=argo,
        model_dz={"model": observations.model_depth_thickness(model, OM4_SPEC)},
        window=(pd.Timestamp("2021-01-01"), pd.Timestamp("2022-12-31")),
        bootstrap_samples=0,
    )

    primary = frame[frame["period_kind"] == "primary_complete_years"]
    assert not primary.empty
    assert primary["n_paired_cells"].notna().all(), (
        "n_paired_cells is declared in COLUMNS but never populated"
    )
    assert primary["paired_ocean_fraction"].notna().all()


def test_scalar_and_map_detrending_agree_on_a_non_uniform_axis():
    """The two detrending paths must not disagree; the code says so explicitly.

    `detrend_linear_dataarray` regresses on elapsed days, and its comment gives
    the reason: "The two agree only on a uniform axis, and the axis is not
    uniform once `monthly_mean_of_complete_months` drops a month -- at which
    point an index regression silently treats the gap as if no time had passed.
    `series_linear_trend_per_year` already uses elapsed time; these two must not
    disagree."

    `series_without_linear_trend` -- the one `series_residual_variance` actually
    calls -- regresses on `np.arange(len(values))`. So on the very axis that
    comment describes, the map path and the scalar path remove different trends
    from the same data. Here the series is linear in elapsed time, so both
    residuals must be zero; the index regression leaves a sawtooth at the gap
    and reports it as residual variance the model failed to reproduce.
    """
    months = pd.DatetimeIndex(
        [
            stamp
            for stamp in pd.date_range("2021-01-01", "2022-12-01", freq="MS")
            # The gap `monthly_mean_of_complete_months` is documented to create.
            if stamp.month not in (5, 6)
        ]
    )
    elapsed_years = (months - months[0]).days.to_numpy(dtype=float) / 365.25
    values = 2.0 + 0.5 * elapsed_years

    gridded = kernels.detrend_linear_dataarray(
        xr.DataArray(values, dims=["time"], coords={"time": months})
    )
    scalar = kernels.series_without_linear_trend(pd.Series(values, index=months))

    assert float(np.abs(gridded.values).max()) == pytest.approx(0.0, abs=1e-9)
    assert float(np.abs(scalar.to_numpy()).max()) == pytest.approx(0.0, abs=1e-9)


def test_a_grid_one_cell_short_of_global_is_not_treated_as_periodic():
    """`_spans_globe`'s tolerance is a whole grid cell, so it wraps regional grids.

    `_wrap_lon` promises that "regional grids are returned untouched: their
    edges are real boundaries, and inventing data beyond them would be worse
    than dropping a column." The test it rests on is

        isclose(values[-1] - values[0] + spacing, 360.0, atol=abs(spacing))

    whose absolute tolerance is one full cell -- exactly the resolution at which
    "global" and "one cell short of global" stop being distinguishable. A basin
    that stops one column short is declared periodic, its western edge is padded
    with data from its eastern edge, and `_differentiate_lon` then takes a
    derivative across a boundary that does not exist.

    The regional grids in `test_partial_edge_months_and_seam_are_handled` span a
    fraction of the globe, so they exercise `_wrap_lon`'s pass-through but never
    the predicate that decides it. This pins the predicate at its boundary.
    """
    spacing = 5.0
    regional = np.arange(0.0, 351.0, spacing)  # 0..350: a full cell short of 360
    assert not kernels._spans_globe(regional)

    field = xr.DataArray(
        np.cos(np.deg2rad(regional))[None, :],
        dims=["lat", "lon"],
        coords={"lat": [30.0], "lon": regional},
    )
    assert kernels._wrap_lon(field, "lon").equals(field)


def test_a_year_that_keeps_half_its_samples_is_not_a_complete_year():
    """ "Complete" is judged against the record's own cadence, with no floor.

    `require_complete_calendar_years` exists so that "equal-year blocks are what
    make the primary score and its bootstrap comparable" -- a ragged year has to
    be an error "rather than a silently down-weighted block".

    But completeness is decided by `largest_gap <= 2.5 * median_step`, and that
    median is taken over the whole record. A year that drops every second sample
    has gaps of exactly two median steps, so it passes -- and then carries the
    same weight in the score and in the block bootstrap as a year holding twice
    the data. That is the silent down-weighting, unchanged.

    `test_a_calendar_year_missing_a_fortnight_is_not_complete` pins the case the
    gap threshold does catch -- data missing in one lump. Thinning is the case
    it cannot see, because thinning never produces a large gap.
    """
    dense = pd.date_range("2021-01-03 12:00", "2021-12-29 12:00", freq="5D")
    halved = pd.date_range("2022-01-03 12:00", "2022-12-29 12:00", freq="5D")[::2]
    time = dense.append(halved)

    years = kernels.complete_calendar_years(
        xr.DataArray(time, dims=["time"], coords={"time": time})
    )
    assert years == [2021], f"2022 holds {len(halved)} of {len(dense)} samples"
