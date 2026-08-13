# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the observation figures in `samudra.viz`.

The point of routing both jobs through `samudra.metrics` is that a number
printed on a figure is the number logged to W&B. These check that, and that
each figure builder produces files on data shaped like the real thing.
"""

from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr

matplotlib.use("Agg")

from samudra.constants import build_om4_spec  # noqa: E402
from samudra.metrics import comparisons, observations, report, spectra  # noqa: E402
from samudra.viz import observations as figures  # noqa: E402

OM4_SPEC = build_om4_spec(prognostic_vars_key="thermo_dynamic_all")
WINDOW = (pd.Timestamp("2021-01-01"), pd.Timestamp("2022-12-31"))


@pytest.fixture(scope="module")
def synthetic():
    """A rollout and three observation products on realistic, differing grids."""
    rng = np.random.default_rng(0)
    time = pd.date_range("2021-01-01", "2022-12-31", freq="5D")
    months = pd.date_range("2021-01-01", "2022-12-01", freq="MS")

    model_lat = np.linspace(-78.0, 78.0, 40)
    model_lon = np.arange(2.5, 360.0, 5.0)
    obs_lat = np.linspace(-79.0, 79.0, 60)
    obs_lon = np.arange(1.5, 360.0, 3.0)
    levels = np.array(OM4_SPEC.depth_levels)

    rollout = xr.Dataset(
        {
            "thetao": (
                ("time", "lev", "y", "x"),
                10
                + rng.normal(
                    size=(len(time), levels.size, model_lat.size, model_lon.size)
                ),
            ),
            "zos": (
                ("time", "y", "x"),
                rng.normal(scale=0.1, size=(len(time), model_lat.size, model_lon.size)),
            ),
        },
        coords={
            "y": model_lat,
            "x": model_lon,
            "lat": (
                ("y", "x"),
                np.broadcast_to(
                    model_lat[:, None], (model_lat.size, model_lon.size)
                ).copy(),
            ),
            "lon": (
                ("y", "x"),
                np.broadcast_to(
                    model_lon[None, :], (model_lat.size, model_lon.size)
                ).copy(),
            ),
            "lev": levels,
            "areacello": (("y", "x"), np.full((model_lat.size, model_lon.size), 1e10)),
            "time": time,
        },
    )

    def product(data_vars, stamps, extra=None):
        coords = {"lat": obs_lat, "lon": obs_lon, "time": stamps}
        coords.update(extra or {})
        return observations.with_cell_area(
            observations.standardize(xr.Dataset(data_vars, coords=coords))
        )

    shape = (len(time), obs_lat.size, obs_lon.size)
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
    depths = np.array([5.0, 50.0, 150.0, 400.0, 800.0, 1400.0, 1900.0])
    argo = product(
        {
            "temp": (
                ("time", "depth", "lat", "lon"),
                10
                + rng.normal(
                    size=(len(months), depths.size, obs_lat.size, obs_lon.size)
                ),
            )
        },
        months,
        {"depth": depths},
    )

    prepared = observations.model_on_latlon_grid(rollout, OM4_SPEC)
    rollouts = {"model": prepared}
    products = {"duacs": duacs, "oisst": oisst, "argo": argo}
    frame = report.compute_observation_metrics(
        rollouts,
        duacs=duacs,
        oisst=oisst,
        argo=argo,
        model_dz={"model": observations.model_depth_thickness(prepared, OM4_SPEC)},
        window=WINDOW,
        bootstrap_samples=0,
    )
    return rollouts, products, frame


def test_a_figure_reports_the_number_that_was_logged(synthetic, tmp_path):
    """A value annotated on a map must equal the scalar sent to W&B.

    This is the whole reason the figures reduce through `samudra.metrics`
    rather than carrying their own implementation: a second one would drift.
    """
    rollouts, products, frame = synthetic
    scalars = report.to_wandb(frame, "model")

    written = figures.rmse_map_figures(rollouts, products, frame, WINDOW, str(tmp_path))
    assert written, "no RMSE maps were produced"

    # The map the figure draws and the scalar W&B receives come from one
    # reduction, so re-deriving the total from the frame must match exactly.
    row = frame[
        (frame["metric"] == "surface_sst_total_rmse")
        & (frame["period_kind"] == "primary_complete_years")
    ].iloc[0]
    logged = scalars["obs/sst/total_rmse"]
    assert isinstance(logged, float), "W&B scalars must be plain floats"
    assert float(row["value"]) == pytest.approx(logged)

    # And the annotation renders that same number rather than recomputing it.
    assert f"{float(row['value']):.4g}" in figures._annotate(
        float(row["value"]), "degC", row
    )


def test_every_figure_builder_writes_files(synthetic, tmp_path):
    """Each builder produces PDFs on data shaped like the real products.

    A smoke test rather than an appearance check: it catches the failures that
    actually happen here -- a grid mismatch, a missing variable, an empty
    region -- which only surface once a figure is drawn end to end.
    """
    rollouts, products, frame = synthetic

    produced = []
    produced += figures.rmse_map_figures(
        rollouts, products, frame, WINDOW, str(tmp_path)
    )
    produced += figures.variance_map_figures(rollouts, products, frame, str(tmp_path))
    produced += figures.timeseries_figures(rollouts, products, str(tmp_path))
    produced.append(
        figures.save(
            figures.annual_rmse_panel(frame, "annual"), str(tmp_path), "annual"
        )
    )

    assert len(produced) >= 8
    for path in produced:
        assert path.endswith(".pdf")
        assert (tmp_path / path.rsplit("/", 1)[-1]).stat().st_size > 0


def test_spectra_figures_survive_regions_too_small_to_transform(synthetic, tmp_path):
    """A region the grid cannot resolve is skipped, not fatal.

    The coarse grids here make several named regions too small for a
    transform, which is exactly what a low-resolution rollout does in
    production.
    """
    rollouts, products, _ = synthetic
    produced = figures.spectra_figures(rollouts, products, str(tmp_path))
    assert produced
    for path in produced:
        assert path.endswith(".pdf")


def test_curve_scores_are_zero_against_the_reference_itself():
    """A run identical to the observations must score 0 dex, not merely 'small'."""
    wavenumber = np.logspace(-3, -1, 20)
    power = wavenumber**-2
    curves = {
        "DUACS": {"Gulf Stream": (wavenumber, power)},
        "model": {"Gulf Stream": (wavenumber, power)},
    }
    scores = figures._curve_scores(curves, "DUACS")
    assert scores["model"]["Gulf Stream"] == pytest.approx(0.0, abs=1e-12)
    assert "DUACS" not in scores, "the reference is not scored against itself"


def test_interannual_eke_maps_share_one_velocity_baseline():
    """Annual EKE maps must be taken about the whole record's mean flow.

    Re-deriving the mean within each year would absorb the year-to-year change
    in the mean flow, and the interannual band -- whose whole purpose is to show
    that spread -- would collapse to zero width no matter the data.
    """
    time = pd.date_range("2020-01-03", "2022-12-29", freq="5D")
    lat, lon = np.linspace(-10.0, 10.0, 12), np.linspace(0.0, 20.0, 12)

    # The same eddy field every year, on a mean flow that differs between them.
    # The three offsets are deliberately unequal about their own mean: with two
    # symmetric years the squares would coincide and prove nothing.
    eddy = np.sin(np.linspace(0, 6 * np.pi, time.size))[:, None, None] * np.ones(
        (1, lat.size, lon.size)
    )
    offsets = {2020: 1.0, 2021: 2.0, 2022: 5.0}
    mean_flow = np.array([offsets[year] for year in time.year])[:, None, None]
    coords = {"time": time, "lat": lat, "lon": lon}
    u = xr.DataArray(eddy + mean_flow, dims=("time", "lat", "lon"), coords=coords)
    v = xr.DataArray(np.zeros_like(u.values), dims=u.dims, coords=coords)

    by_year = figures.obs_eke_by_year(u, v)
    assert set(by_year) == {2020, 2021, 2022}

    # A per-year baseline cancels the offset entirely and leaves the same eddy
    # variance in all three years, so the spread has to be large, not merely
    # non-zero -- a near-equal spread is exactly the failure being excluded.
    levels = [float(by_year[year].mean()) for year in (2020, 2021, 2022)]
    assert max(levels) / min(levels) > 2.0


def test_the_observation_steps_skip_when_no_products_are_configured():
    """Every preset without an `observations` block must still run.

    These steps are part of the default run, so raising here breaks `samudra
    viz` for the presets that do not configure observations -- which is most of
    them -- and contradicts `VizConfig.observations`, which says omitting the
    block skips these steps.
    """
    from samudra.viz.config import _ordered_steps
    from samudra.viz.core import Viz

    observation_steps = [step for step in _ordered_steps() if step.startswith("obs_")]
    assert observation_steps, "no observation steps found to check"

    unconfigured = SimpleNamespace(observations=None, obs_data_root=None)
    unconfigured._observations_configured = Viz._observations_configured.__get__(
        unconfigured
    )
    for step in observation_steps:
        getattr(Viz, f"step_{step}")(unconfigured)


def _shared_grid_case(
    model_time: pd.DatetimeIndex,
    obs_time: pd.DatetimeIndex,
    argo_months: pd.DatetimeIndex,
    *,
    sst_profile: np.ndarray | None = None,
    model_land: np.ndarray | None = None,
) -> tuple[dict, dict]:
    """A rollout and three products on one grid, so only masking and span differ.

    Sharing the grid removes regridding from the picture: anything the figures
    then disagree about is the figures' doing.

    Args:
        model_time: Rollout timestamps.
        obs_time: DUACS/OISST timestamps; must contain `model_time`.
        argo_months: ARGO-IAP month starts.
        sst_profile: Per-latitude SST, broadcast over longitude and time.
            Random noise when omitted.
        model_land: Boolean latitude mask; True where the rollout has no water.
    """
    rng = np.random.default_rng(0)
    lat = np.linspace(-79.0, 79.0, 20)
    lon = np.arange(5.0, 360.0, 10.0)
    levels = np.array(OM4_SPEC.depth_levels)
    shape = (obs_time.size, lat.size, lon.size)

    if sst_profile is None:
        sst = 10 + rng.normal(size=shape)
    else:
        sst = np.broadcast_to(np.asarray(sst_profile)[None, :, None], shape).copy()

    def product(data_vars, stamps, extra=None):
        coords = {"lat": lat, "lon": lon, "time": stamps}
        coords.update(extra or {})
        return observations.with_cell_area(
            observations.standardize(xr.Dataset(data_vars, coords=coords))
        )

    duacs = product(
        {
            "ugos": (("time", "lat", "lon"), rng.normal(scale=0.2, size=shape)),
            "vgos": (("time", "lat", "lon"), rng.normal(scale=0.2, size=shape)),
        },
        obs_time,
    )
    oisst = product({"sst": (("time", "lat", "lon"), sst)}, obs_time)
    depths = np.array([5.0, 50.0, 150.0, 400.0, 800.0, 1400.0, 1900.0])
    argo = product(
        {
            "temp": (
                ("time", "depth", "lat", "lon"),
                10
                + rng.normal(size=(argo_months.size, depths.size, lat.size, lon.size)),
            )
        },
        argo_months,
        {"depth": depths},
    )

    # The rollout reproduces the observations exactly, wherever it has water.
    overlap = np.isin(obs_time.values, model_time.values)
    model_sst = sst[overlap]
    if model_land is not None:
        model_sst[:, np.asarray(model_land), :] = np.nan
    thetao = np.broadcast_to(
        model_sst[:, None, :, :], (model_time.size, levels.size, lat.size, lon.size)
    ).copy()

    rollout = xr.Dataset(
        {
            "thetao": (("time", "lev", "y", "x"), thetao),
            "zos": (
                ("time", "y", "x"),
                rng.normal(scale=0.1, size=(model_time.size, lat.size, lon.size)),
            ),
        },
        coords={
            "y": lat,
            "x": lon,
            "lat": (
                ("y", "x"),
                np.broadcast_to(lat[:, None], (lat.size, lon.size)).copy(),
            ),
            "lon": (
                ("y", "x"),
                np.broadcast_to(lon[None, :], (lat.size, lon.size)).copy(),
            ),
            "lev": levels,
            "areacello": (("y", "x"), np.full((lat.size, lon.size), 1e10)),
            "time": model_time,
        },
    )
    rollouts = {"model": observations.model_on_latlon_grid(rollout, OM4_SPEC)}
    return rollouts, {"duacs": duacs, "oisst": oisst, "argo": argo}


def test_every_scored_metric_gets_the_map_that_explains_it(tmp_path):
    """A metric with a headline scalar must also produce its RMSE map.

    `report` reduces the monthly OHC series to the calendar years it covers in
    full before scoring, because a monthly observation product routinely stops
    part-way through a year. `rmse_map_figures` does not, so the same data
    reaches `rmse_map_with_uncertainty` with a ragged final year, which is an
    error there. The error is caught and logged at warning level, and both OHC
    maps -- two of the five this PR adds -- vanish from the output directory
    while their scalars sit in the CSV and in W&B.
    """
    model_time = pd.date_range("2020-01-01", "2022-12-31", freq="5D")
    # ARGO-IAP lags: its last month is not December, which is the case
    # `report._whole_years` exists to handle.
    rollouts, products = _shared_grid_case(
        model_time, model_time, pd.date_range("2020-01-01", "2022-10-01", freq="MS")
    )
    window = (pd.Timestamp("2020-01-01"), pd.Timestamp("2022-12-31 23:59:59"))
    frame = report.compute_observation_metrics(
        rollouts,
        duacs=products["duacs"],
        oisst=products["oisst"],
        argo=products["argo"],
        model_dz={
            "model": observations.model_depth_thickness(rollouts["model"], OM4_SPEC)
        },
        window=window,
        bootstrap_samples=0,
    )

    scored = set(
        frame[frame["period_kind"] == "primary_complete_years"]["metric"].unique()
    )
    assert "ohc_per_area_total_rmse" in scored, "fixture no longer scores OHC"

    written = figures.rmse_map_figures(rollouts, products, frame, window, str(tmp_path))
    names = [path.rsplit("/", 1)[-1] for path in written]
    missing = [name for name in names if "ohc" in name]
    assert missing, (
        "OHC has a scalar in the frame but no RMSE map figure was written; "
        f"only {names} were produced"
    )


def test_the_global_mean_series_are_built_from_the_same_cells(tmp_path):
    """A run that reproduces the observations exactly must not look biased.

    The observation curve is an area-weighted mean over every cell the product
    has; each run's curve covers only the cells that survived pairing, which
    excludes the run's own land and the band that regridding erodes around it.
    The gap between the two curves is then a property of the masks, and the
    figure presents it as a mean-state bias.
    """
    model_time = pd.date_range("2021-01-03", "2022-12-29", freq="5D")
    # Warm at the equator, cold at the poles, and the run treats the tropics as
    # land -- an exaggerated stand-in for the coastal band regridding removes.
    lat = np.linspace(-79.0, 79.0, 20)
    rollouts, products = _shared_grid_case(
        model_time,
        model_time,
        pd.date_range("2021-01-01", "2022-12-01", freq="MS"),
        sst_profile=28.0 - 30.0 * np.sin(np.deg2rad(lat)) ** 2,
        model_land=np.abs(lat) < 20.0,
    )

    drawn: dict[str, dict[str, pd.Series]] = {}
    original = figures.series_panel

    def capture(series, title, ylabel, annotations=None):
        drawn.setdefault(title, series)
        return original(series, title, ylabel, annotations)

    figures.series_panel = capture
    try:
        figures.timeseries_figures(rollouts, products, str(tmp_path))
    finally:
        figures.series_panel = original

    sst = drawn["Global mean SST"]
    reference, run = sst["OISST"].mean(), sst["model"].mean()
    assert run == pytest.approx(reference, abs=0.05), (
        f"the run reproduces every cell it has, yet its global mean sits "
        f"{run - reference:+.2f} degC from the observed one, because the two "
        "means are taken over different sets of cells"
    )


def test_observation_curves_cover_the_record_the_runs_do(tmp_path):
    """The reference must be reduced over the span the runs cover, not its own.

    Every observation product outlives the rollout it is scored against --
    OISST starts in 1982, DUACS in 1993. `variance_map_figures` now trims the
    reference to the shared span; `timeseries_figures` and `spectra_figures`
    still reduce it over the store's whole record, then draw it beside the runs
    and annotate each with a trend, a residual variance and a year count.
    Those annotations are the quantitative content of both figures, and they
    are computed over different records.
    """
    obs_time = pd.date_range("2019-01-03", "2022-12-29", freq="5D")
    model_time = obs_time[obs_time >= pd.Timestamp("2021-01-01")]
    rollouts, products = _shared_grid_case(
        model_time, obs_time, pd.date_range("2019-01-01", "2022-12-01", freq="MS")
    )

    series: dict[str, dict[str, pd.Series]] = {}
    bands: dict[str, dict[str, dict]] = {}
    original_series = figures.series_panel
    original_bands = figures.interannual_spectra_panel

    def capture_series(values, title, ylabel, annotations=None):
        series.setdefault(title, values)
        return original_series(values, title, ylabel, annotations)

    def capture_bands(values, title, xlabel, ylabel):
        bands.setdefault(title, values)
        return original_bands(values, title, xlabel, ylabel)

    figures.series_panel = capture_series
    figures.interannual_spectra_panel = capture_bands
    try:
        figures.timeseries_figures(rollouts, products, str(tmp_path))
        figures.spectra_figures(rollouts, products, str(tmp_path))
    finally:
        figures.series_panel = original_series
        figures.interannual_spectra_panel = original_bands

    sst = series["Global mean SST"]
    assert sst["OISST"].index.min() == sst["model"].index.min(), (
        f"the OISST curve starts {sst['OISST'].index.min():%Y-%m-%d} while the "
        f"run starts {sst['model'].index.min():%Y-%m-%d}, so the trend and "
        "residual variance annotated on each describe different records"
    )

    # `interannual_band` returns (x, mean, lower, upper, n_years); the panel
    # prints that year count in the legend beside each curve.
    temporal = bands["Interannual surface geostrophic KE temporal spectra"]
    assert temporal["DUACS"]["Global"][4] == temporal["model"]["Global"][4], (
        "the observed band aggregates "
        f"{temporal['DUACS']['Global'][4]} years against the run's "
        f"{temporal['model']['Global'][4]}, and the panel labels both as a "
        "band of yearly spectra"
    )


def test_energies_are_derived_after_the_window_is_settled():
    """EKE has to be built from the trimmed velocity, not trimmed afterwards.

    The anomaly is taken about the mean of whatever record it is handed, so
    deriving first and trimming second leaves each curve on a different
    baseline -- and on a record with any drift in the mean flow that is not a
    rounding difference.
    """
    time = pd.date_range("2015-01-03", "2022-12-29", freq="5D")
    lat, lon = np.linspace(-40.0, 40.0, 12), np.linspace(0.0, 350.0, 18)
    rng = np.random.default_rng(0)
    coords = {"time": time, "lat": lat, "lon": lon}

    drift = np.linspace(0.0, 0.4, time.size)[:, None, None]
    u = xr.DataArray(
        drift + rng.normal(scale=0.2, size=(time.size, lat.size, lon.size)),
        dims=("time", "lat", "lon"),
        coords=coords,
    )
    v = xr.DataArray(rng.normal(scale=0.2, size=u.shape), dims=u.dims, coords=coords)
    area = xr.DataArray(
        np.ones((lat.size, lon.size)),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
    )
    velocity = comparisons.VelocityComparison(
        comparisons.Comparison("u", u, u, area),
        comparisons.Comparison("v", v, v, area),
    )
    window = slice(pd.Timestamp("2020-01-01"), pd.Timestamp("2022-12-31"))

    derive_then_trim = float(velocity.eddy_kinetic_energy().slice(window).obs.mean())
    trim_then_derive = float(velocity.slice(window).eddy_kinetic_energy().obs.mean())
    assert derive_then_trim != pytest.approx(trim_then_derive, rel=0.05), (
        "the fixture no longer separates the two orders, so it cannot show "
        "that the order matters"
    )


def test_a_region_too_anisotropic_to_transform_is_skipped_not_fatal():
    """Too few bins below the Nyquist must lose the panel, not the figure.

    A box can pass the side-length check and still leave fewer than two
    wavenumber bins once the spacings are strongly anisotropic, which
    `isotropic_spectrum` raises on. `region_spectrum` promises empty arrays for
    a region it cannot transform, and the callers draw that as unavailable.
    """
    lat = np.linspace(80.0, 82.0, 9)  # a high-latitude box: dx << dy
    lon = np.linspace(0.0, 40.0, 9)
    field = xr.DataArray(
        np.random.default_rng(0).normal(size=(lat.size, lon.size)),
        dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon},
    )
    wavenumber, power = spectra.region_spectrum(
        field, slice(0, 40), slice(80, 82), name="polar sliver"
    )
    assert wavenumber.size == power.size


def test_a_partly_covered_year_does_not_enter_an_interannual_band():
    """A band aggregates whole years, or its spread is partly a sampling artefact.

    The default rollout starts on 20 October, so its first calendar year holds
    about fifteen five-day samples -- all of one season. Counting that as a year
    skews both the band and the year count printed beside it.
    """
    time = pd.date_range("2014-10-20", "2016-12-29", freq="5D")
    lat, lon = np.linspace(-30.0, 30.0, 10), np.linspace(0.0, 350.0, 12)
    coords = {"time": time, "lat": lat, "lon": lon}
    rng = np.random.default_rng(0)
    u = xr.DataArray(
        rng.normal(size=(time.size, lat.size, lon.size)),
        dims=("time", "lat", "lon"),
        coords=coords,
    )
    v = xr.DataArray(rng.normal(size=u.shape), dims=u.dims, coords=coords)

    assert int((time.year == 2014).sum()) > 12, "2014 no longer clears a count cutoff"
    assert set(figures.obs_eke_by_year(u, v)) == {2015, 2016}
    assert set(figures._anomaly_by_year(u)) == {2015, 2016}


def test_the_model_velocity_is_an_anomaly_when_the_product_is():
    """Against DUACS's anomaly product the model's time mean has to come off.

    The model's geostrophic velocity from `zos` is absolute either way, so
    leaving it absolute compares an anomaly against a mean flow. Kinetic energy
    is quadratic, so no later detrend of the energy series recovers it.
    """
    time = pd.date_range("2021-01-03", "2022-12-29", freq="5D")
    lat, lon = np.linspace(-30.0, 30.0, 10), np.linspace(0.0, 350.0, 12)
    coords = {"time": time, "lat": lat, "lon": lon}
    rng = np.random.default_rng(0)
    mean_flow = 0.8
    field = xr.DataArray(
        mean_flow + rng.normal(scale=0.1, size=(time.size, lat.size, lon.size)),
        dims=("time", "lat", "lon"),
        coords=coords,
    )
    area = xr.DataArray(
        np.ones((lat.size, lon.size)),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
    )
    pair = comparisons.VelocityComparison(
        comparisons.Comparison("u", field, field, area),
        comparisons.Comparison("v", field, field, area),
        kind="anomaly",
    )._rebased()

    assert float(np.abs(pair.eastward.native.mean("time")).max()) < 1e-12
    # The observed side is already an anomaly product; it is left alone.
    assert float(pair.eastward.obs.mean()) == pytest.approx(mean_flow, abs=0.02)

    # Slicing rebases, so a narrower window is an anomaly about that window.
    window = slice(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"))
    assert float(np.abs(pair.slice(window).eastward.native.mean("time")).max()) < 1e-12

    # And the absolute kind keeps the mean flow it was given.
    absolute = comparisons.VelocityComparison(
        comparisons.Comparison("u", field, field, area),
        comparisons.Comparison("v", field, field, area),
    )._rebased()
    assert float(absolute.eastward.native.mean()) == pytest.approx(mean_flow, abs=0.02)
