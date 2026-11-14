# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the scorecard (`samudra.metrics.skill`, `samudra.metrics.scorecard`).

Each test pins one property a user relies on: the meaning of skill at its
fixed points, indifference to land and to units, loud failure on misaligned
inputs and on non-finite rollouts, volume weighting of layers, and the
champion comparison and gate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from hypothesis import given, settings
from hypothesis import strategies as st

from samudra.config import ScorecardConfig
from samudra.metrics import skill
from samudra.metrics.scorecard import card, gate, run, series
from samudra.utils.location import UnresolvedLocation

CHANNELS = ["thetao_0", "thetao_1", "so_0", "so_1"]
DEPTH = {"thetao_0": 2.5, "thetao_1": 1000.0, "so_0": 2.5, "so_1": 1000.0}
DZ = {"thetao_0": 5.0, "thetao_1": 1000.0, "so_0": 5.0, "so_1": 1000.0}
YEARS = list(range(2015, 2023))
NLAT, NLON = 4, 6


def _times(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="5D") + pd.Timedelta(hours=12)


def _field(times, rng, offset: float, land: np.ndarray) -> xr.DataArray:
    doy = np.asarray(times.dayofyear)
    seasonal = np.sin(2 * np.pi * doy / 365.0)[:, None, None]
    noise = rng.normal(size=(len(times), NLAT, NLON))
    data = offset + 2.0 * seasonal + 0.3 * np.cumsum(noise, axis=0) / 10 + 0.2 * noise
    data[:, land] = np.nan
    return xr.DataArray(data, dims=("time", "lat", "lon"), coords={"time": times})


def _truth_and_clim(seed: int = 0):
    rng = np.random.default_rng(seed)
    land = np.zeros((NLAT, NLON), bool)
    land[0, :2] = True
    area = xr.DataArray(
        np.cos(np.deg2rad(np.linspace(-60, 60, NLAT)))[:, None] * np.ones((1, NLON)),
        dims=("lat", "lon"),
    )
    offsets = {"thetao_0": 15.0, "thetao_1": 4.0, "so_0": 35.0, "so_1": 34.7}
    train_t = _times("2005-01-01", "2010-12-31")
    test_t = _times("2014-10-10", "2022-12-31")
    train = xr.Dataset({c: _field(train_t, rng, offsets[c], land) for c in CHANNELS})
    truth = xr.Dataset({c: _field(test_t, rng, offsets[c], land) for c in CHANNELS})
    truth = truth.assign_coords(areacello=area)
    clim = xr.Dataset({c: skill.pentad_climatology(train[c]) for c in CHANNELS})
    return truth, clim


def _predictions(truth: xr.Dataset, transform) -> xr.Dataset:
    """A 'rollout' starting two steps after the truth begins (as Samudra's does)."""
    future = truth.drop_vars("areacello").isel(time=slice(2, None))
    return xr.Dataset({c: transform(c, future[c]) for c in CHANNELS})


def _card(predictions, truth, clim, samples: int = 200):
    tables = series.compute(predictions, truth, clim, CHANNELS)
    frame = card.build(
        tables, depth=DEPTH, dz=DZ, years=YEARS, bootstrap_samples=samples
    )
    return tables, frame


def _value(frame, metric, *, layer="", channel="", variable=None, period_kind="window"):
    sel = frame[
        (frame.metric == metric)
        & (frame.layer == layer)
        & (frame.channel == channel)
        & (frame.period_kind == period_kind)
    ]
    if variable is not None:
        sel = sel[sel.variable == variable]
    assert len(sel) == 1, sel
    return float(sel.value.iloc[0])


@pytest.fixture(scope="module")
def data():
    return _truth_and_clim()


def test_skill_fixed_points(data):
    """The truth scores skill 1; each reference, used as the forecast, scores 0 against itself."""
    truth, clim = data
    _, perfect = _card(_predictions(truth, lambda c, f: f), truth, clim)
    assert _value(perfect, "skill_vs_climatology", channel="thetao_0") == 1.0
    assert _value(perfect, "rmse", channel="so_1") == 0.0

    as_clim = _predictions(
        truth,
        lambda c, f: skill.climatology_forecast(clim[c], f["time"]).where(
            np.isfinite(f)
        ),
    )
    _, frame = _card(as_clim, truth, clim)
    assert _value(frame, "skill_vs_climatology", channel="thetao_1") == pytest.approx(
        0.0, abs=1e-12
    )

    t0 = series.initial_state(truth, truth.time.values[2])
    as_pers = _predictions(
        truth, lambda c, f: xr.full_like(f, 0) + t0[c].drop_vars("time")
    )
    _, frame = _card(as_pers, truth, clim)
    assert _value(
        frame, "skill_vs_persistence", layer="upper", variable="so"
    ) == pytest.approx(0.0, abs=1e-12)


@settings(deadline=None, max_examples=10)
@given(scale=st.floats(0.01, 100.0), shift=st.floats(-50.0, 50.0))
def test_skill_does_not_depend_on_units(data, scale, shift):
    """Changing a channel's units (scale and offset) leaves every skill score unchanged."""
    truth, clim = data
    rng = np.random.default_rng(1)
    noisy = _predictions(truth, lambda c, f: f + 0.1 * rng.normal(size=f.shape))
    _, base = _card(noisy, truth, clim, samples=0)

    def convert(ds):
        return ds.map(lambda v: v * scale + shift)

    _, moved = _card(
        convert(noisy),
        convert(truth).assign_coords(areacello=truth.areacello),
        convert(clim),
        0,
    )
    for metric in ("skill_vs_climatology", "skill_vs_persistence"):
        assert _value(moved, metric, channel="thetao_0") == pytest.approx(
            _value(base, metric, channel="thetao_0"), rel=1e-6, abs=1e-9
        )


def test_land_values_are_ignored(data):
    """Whatever a model writes on land changes no score."""
    truth, clim = data
    rng = np.random.default_rng(2)
    noise = rng.normal(size=(truth.sizes["time"] - 2, NLAT, NLON))
    base = _predictions(truth, lambda c, f: f + 0.1 * noise)
    junk = base.map(lambda v: v.fillna(1e6))
    _, a = _card(base, truth, clim, samples=0)
    _, b = _card(junk, truth, clim, samples=0)
    rankable = a.better != card.BOUNDED
    pd.testing.assert_series_equal(a.value[rankable], b.value[rankable])


def test_layers_are_volume_weighted(data):
    """A full-depth bias is the thickness-weighted mean of the levels' biases."""
    truth, clim = data
    biased = _predictions(truth, lambda c, f: f + (1.0 if c.endswith("_0") else 0.0))
    _, frame = _card(biased, truth, clim, samples=0)
    expected = (1.0 * DZ["thetao_0"]) / (DZ["thetao_0"] + DZ["thetao_1"])
    assert _value(
        frame, "bias", layer=card.FULL_DEPTH, variable="thetao"
    ) == pytest.approx(expected)


def test_misaligned_inputs_fail_loudly(data):
    truth, clim = data
    good = _predictions(truth, lambda c, f: f)
    with pytest.raises(ValueError, match="timestamps must match"):
        series.compute(
            good.isel(time=slice(0, -1)), truth.isel(time=slice(0, -3)), clim, CHANNELS
        )
    with pytest.raises(ValueError, match="own resolution"):
        series.compute(good.isel(lon=slice(0, 3)), truth, clim, CHANNELS)
    with pytest.raises(ValueError, match="pentad"):
        skill.pentad_climatology(truth["thetao_0"].isel(time=slice(0, 10)))


def test_rollout_in_the_eval_writers_layout(data):
    """The writer saves depth-stacked variables on y/x; they score like flat channels."""
    truth, clim = data
    flat = _predictions(truth, lambda c, f: f + 0.1)
    lat2d = xr.DataArray(np.zeros((NLAT, NLON)), dims=("y", "x"))
    stacked = xr.Dataset(
        {
            base: xr.concat([flat[f"{base}_{i}"] for i in (0, 1)], dim="lev")
            .transpose("time", "lev", "lat", "lon")
            .rename({"lat": "y", "lon": "x"})
            for base in ("thetao", "so")
        }
    ).assign_coords(lat=lat2d, lon=lat2d)
    _, a = _card(flat, truth, clim, samples=0)
    _, b = _card(stacked, truth, clim, samples=0)
    pd.testing.assert_series_equal(a.value, b.value)


def test_nonfinite_rollout_fails_the_gate_and_poisons_its_error(data):
    """A failed step makes the error NaN, never quietly smaller."""
    truth, clim = data

    def blow_up(c, f):
        f = f.copy()
        if c == "thetao_0":
            f[100, 2, 3] = np.nan
        return f

    tables, frame = _card(_predictions(truth, blow_up), truth, clim, samples=0)
    rows = tables.sums[tables.sums.channel == "thetao_0"]
    assert not np.isfinite(rows.se_model.iloc[100])
    assert np.isnan(_value(frame, "rmse", channel="thetao_0"))
    assert np.isnan(_value(frame, "rmse", layer="upper", variable="thetao"))
    results = {c.name: c for c in gate.checks(frame, gate.GatePolicy(), None)}
    assert not results["finite"].passed
    assert results["salinity_nonnegative"].passed


def test_sst_runaway_fails_the_gate(data):
    """The #969 failure mode: a warm surface runaway is caught without a champion."""
    truth, clim = data
    hot = _predictions(truth, lambda c, f: f + (5.0 if c == "thetao_0" else 0.0))
    _, frame = _card(hot, truth, clim, samples=0)
    results = {c.name: c for c in gate.checks(frame, gate.GatePolicy(), None)}
    assert not results["sst_max"].passed
    assert results["finite"].passed


def _tables(predictions, truth, clim):
    tables = series.compute(predictions, truth, clim, CHANNELS)
    return card.year_tables(tables, depth=DEPTH, dz=DZ, years=YEARS)


def test_champion_comparison(data):
    """Identical models never differ; a uniformly worse one regresses on every gated row."""
    truth, clim = data
    rng = np.random.default_rng(3)
    noise = rng.normal(size=(truth.sizes["time"] - 2, NLAT, NLON))
    champ = _tables(_predictions(truth, lambda c, f: f + 0.05 * noise), truth, clim)
    same = gate.compare(champ, champ, bootstrap_samples=200, level=0.99)
    assert set(same.verdict) == {gate.INDISTINGUISHABLE}

    worse_preds = _predictions(truth, lambda c, f: f + 0.5 * noise + 0.3)
    worse = _tables(worse_preds, truth, clim)
    cmp = gate.compare(worse, champ, bootstrap_samples=200, level=0.99)
    _, frame = _card(worse_preds, truth, clim, samples=0)
    results = {c.name: c for c in gate.checks(frame, gate.GatePolicy(), cmp)}
    assert not results["no_regression_vs_champion"].passed

    with pytest.raises(ValueError, match="same window"):
        gate.compare(
            worse,
            {k: v.iloc[1:] for k, v in champ.items()},
            bootstrap_samples=10,
            level=0.9,
        )
    with pytest.raises(ValueError, match="bootstrap_samples"):
        gate.compare(worse, champ, bootstrap_samples=0, level=0.9)


def test_comparison_ranks_window_skill_not_mean_of_yearly_skill():
    """Skill is a ratio of means: a model worse over the window is never an improvement.

    Per-year skill differences would weight quiet years (small reference error)
    far beyond their share, and here would call the worse model better.
    """
    years = pd.Index(YEARS)
    ref = np.tile([10.0, 1.0], 4)
    base = {"bias": 0.0, "std_ratio": 1.0, "se_persistence": ref}

    def table(se_model):
        return {
            ("layer", "thetao", "upper", ""): pd.DataFrame(
                {"se_model": se_model, "se_climatology": ref, **base}, index=years
            )
        }

    champion = table(0.5 * ref)
    candidate = table(np.tile([5.5, 0.1], 4))
    cmp = gate.compare(candidate, champion, bootstrap_samples=2000, level=0.95)
    row = cmp[cmp.metric == "skill_vs_climatology"].iloc[0]
    assert row.value < row.value_champion
    assert row.badness_diff > 0
    assert row.verdict != gate.IMPROVEMENT


def test_partial_years_are_rejected(data):
    """An equal-weight year block must be a full year."""
    truth, clim = data
    short = _predictions(truth, lambda c, f: f).sel(time=slice(None, "2022-06-30"))
    tables = series.compute(short, truth, clim, CHANNELS)
    with pytest.raises(ValueError, match="too few steps"):
        card.build(tables, depth=DEPTH, dz=DZ, years=YEARS, bootstrap_samples=0)


def test_scorecard_directory_round_trips(data, tmp_path):
    """A scorecard reads back identically; mismatched schema or setup is refused."""
    truth, clim = data
    cfg = ScorecardConfig(
        track="test",
        climatology_location=UnresolvedLocation(path="clim.zarr"),
        bootstrap_samples=50,
        champion=str(tmp_path),
    )
    tables = series.compute(
        _predictions(truth, lambda c, f: f + 0.01), truth, clim, CHANNELS
    )
    frame = card.build(tables, depth=DEPTH, dz=DZ, years=YEARS, bootstrap_samples=50)
    meta = {
        "schema_version": run.SCHEMA_VERSION,
        "track": "test",
        "label": "x",
        "window": "2015-2022",
        "truth": "synthetic",
        "first_forecast": "2014-10-20",
        "n_steps": 597,
        "climatology_period": "2005 to 2010",
    }
    scored = run.Scored(frame, meta, gate.checks(frame, run.policy(cfg), None), None)
    run.write(tmp_path, tables, scored)
    back, _ = run.read_scorecard(str(tmp_path))
    pd.testing.assert_frame_equal(back, frame)
    assert (tmp_path / "scorecard.md").read_text().startswith("# Scorecard: x")

    with pytest.raises(ValueError, match="first_forecast"):
        run._require_comparable({**meta, "first_forecast": "2015-01-05"}, meta)

    (tmp_path / run.META_FILE).write_text('{"schema_version": 0}')
    with pytest.raises(ValueError, match="schema"):
        run.read_scorecard(str(tmp_path))
