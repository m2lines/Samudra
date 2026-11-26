# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Observation preparation, checked against physics and time-averaging identities.

Products are synthetic, so each oracle is exact: a uniform column's heat content
is rho c_p T times the band's thickness; a day's mean flux times one day is the
energy that the day's hourly accumulations add up to; a field linear in time
averages, over any window, to its value at the window's midpoint. Where a test
runs `prepare`, source and target grids are the same, so the remap is the
identity and only the step under test can change a value.
"""

import argparse
import tempfile
from pathlib import Path

import gsw
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from hypothesis import assume, given
from hypothesis import strategies as st
from ocean_preprocessing.vertical_grid import OM4_DEPTH_LEVELS, OM4_LAYER_THICKNESS

from samudra.constants import RHO_0
from samudra.invariants import geometry
from samudra.metrics.kernels import CP
from samudra.observations.archive import ObservationArchive, trajectory
from samudra.observations.constants import OBSERVED_DEPTHS, STEP_DAYS
from samudra.observations.prepare import FORCING, interpolate_depth, prepare
from tests import requires_xesmf
from tests.ocean_strategies import OM4_INTERFACES

pytestmark = pytest.mark.invariant(status="agreed")

# IAP's 41 standard levels (m), as in argo-iap.zarr.
IAP_DEPTHS = np.array(
    [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200]
    + list(range(250, 1000, 50))
    + list(range(1000, 1900, 100))
    + [2000],
    dtype=float,
)
# A small global grid: 3 latitude bands, 4 longitude sectors.
LAT, LAT_B = np.array([-60.0, 0.0, 60.0]), np.array([-90.0, -30.0, 30.0, 90.0])
LON, LON_B = np.array([45.0, 135.0, 225.0, 315.0]), np.arange(0.0, 361.0, 90.0)
OM4_TARGET_DEPTHS = OM4_DEPTH_LEVELS[:OBSERVED_DEPTHS]


def run_prepare(root: Path, store: str, dataset: xr.Dataset, product: str, **kw):
    """Write a product store and prepare it onto the same grid."""
    dataset.assign_coords(lat_b=LAT_B, lon_b=LON_B).to_zarr(
        root / "products" / f"{store}.zarr", consolidated=True
    )
    np.savez(root / "grid.npz", lat=LAT, lon=LON, lat_b=LAT_B, lon_b=LON_B)
    prepare(
        argparse.Namespace(
            source=root / "products",
            grid=root / "grid.npz",
            output=root / "daily",
            source_grid=None,
            product=product,
            year=2001,
            limit=kw.get("limit", 0),
        )
    )
    return root / "daily" / product / "2001"


def iap_store(temp: np.ndarray, salt: np.ndarray, depth: np.ndarray) -> xr.Dataset:
    """One month in argo-iap.zarr's layout: (time, lat, lon, depth_std)."""
    dims = ("time", "lat", "lon", "depth_std")
    return xr.Dataset(
        {
            "temp": (dims, temp.transpose(1, 2, 0)[None].astype(np.float32)),
            "salt": (dims, salt.transpose(1, 2, 0)[None].astype(np.float32)),
        },
        coords={
            "time": [pd.Timestamp("2001-03-01")],
            "lat": LAT,
            "lon": LON,
            "depth_std": depth,
        },
    )


@requires_xesmf
@given(
    st.floats(-1.8, 30.0),
    st.floats(-0.02, 0.02),
    st.lists(
        st.tuples(st.integers(0, 3), st.integers(0, len(IAP_DEPTHS))),
        min_size=12,
        max_size=12,
    ),
)
def test_iap_heat_content_is_rho_cp_t_dz_over_fully_observed_bands(
    t0, gradient, columns
):
    """Each band's heat content is rho0 c_p times the integral of T over it, if the band is fully observed.

    For a profile linear in depth, T = t0 + gradient * z, the integral is exact
    when each level stands for the water closer to it than to its neighbours.
    Each column is observed from level ``top`` to level ``bottom``; a band
    (0-700 m, 700-2000 m) needs every IAP level inside it. The constants are
    `samudra.metrics`', so the observed heat content compares with a model's.
    """
    observed = np.zeros((len(IAP_DEPTHS), 3, 4), dtype=bool)
    for cell, (top, bottom) in enumerate(columns):
        observed[top:bottom, cell // 4, cell % 4] = True
    profile = (t0 + gradient * IAP_DEPTHS)[:, None, None]
    temp = np.where(observed, profile, np.nan)
    with tempfile.TemporaryDirectory() as root:
        out = run_prepare(
            Path(root), "argo-iap", iap_store(temp, temp * 0 + 35.0, IAP_DEPTHS), "iap"
        )
        with np.load(out / "2001-03.npz") as shard:
            ohc = shard["ohc"]
    for band, (low, high) in enumerate([(0, 700), (700, 2000)]):
        inside = (IAP_DEPTHS >= low) & (IAP_DEPTHS <= high)
        complete = observed[inside].all(axis=0)
        np.testing.assert_array_equal(np.isfinite(ohc[band]), complete)
        integral = t0 * (high - low) + gradient * (high**2 - low**2) / 2
        # Within the heat of moving the profile 1 cm, plus float32 round-off.
        tolerance = (
            RHO_0 * CP * (abs(gradient) * (high - low) * 0.01 + 1e-6 * 30 * high)
        )
        np.testing.assert_allclose(
            ohc[band][complete], RHO_0 * CP * integral, rtol=0, atol=tolerance
        )


@requires_xesmf
@given(st.integers(0, 2**32 - 1))
def test_iap_salinity_is_converted_with_each_cells_own_position_and_pressure(seed):
    """IAP stores absolute salinity; each output cell is TEOS-10 practical salinity at its own lat, lon and depth."""
    rng = np.random.default_rng(seed)
    shape = (OBSERVED_DEPTHS, len(LAT), len(LON))
    absolute = rng.uniform(30.0, 38.0, shape)
    with tempfile.TemporaryDirectory() as root:
        out = run_prepare(
            Path(root),
            "argo-iap",
            iap_store(np.full(shape, 10.0), absolute, OM4_TARGET_DEPTHS),
            "iap",
        )
        with np.load(out / "2001-03.npz") as shard:
            practical = shard["values"][1]
    expected = np.empty(shape)
    for k, z in enumerate(OM4_TARGET_DEPTHS):
        for i, lat in enumerate(LAT):
            for j, lon in enumerate(LON):
                pressure = gsw.p_from_z(-z, lat)
                expected[k, i, j] = gsw.SP_from_SA(
                    np.float32(absolute[k, i, j]), pressure, lon, lat
                )
    np.testing.assert_allclose(practical, expected, rtol=0, atol=1e-5)


@requires_xesmf
@given(st.integers(0, 2**32 - 1))
def test_era5_daily_means_keep_each_days_energy_and_water(seed):
    """Daily flux x 86400 s equals the day's accumulated total; states average the day's hours.

    ERA5 accumulations at hour h cover (h - 1h, h], so day D gathers D 01:00
    through D+1 00:00. Random hourly values make an hour's shift visible.
    """
    hours = pd.date_range("2001-03-01", "2001-03-04", freq="h")
    rng = np.random.default_rng(seed)
    hourly = {name: rng.uniform(0.0, 1e6, len(hours)) for name in FORCING}
    dataset = xr.Dataset(
        {
            name: (
                ("time", "lat", "lon"),
                np.broadcast_to(values[:, None, None], (len(hours), 3, 4)).astype(
                    np.float32
                ),
            )
            for name, values in hourly.items()
        },
        coords={"time": hours, "lat": LAT, "lon": LON},
    )
    with tempfile.TemporaryDirectory() as root:
        out = run_prepare(Path(root), "era5-surface", dataset, "era5", limit=3)
        days = pd.date_range("2001-03-01", periods=3)
        shards = {}
        for day in days:
            with np.load(out / f"{day:%Y-%m-%d}.npz") as shard:
                shards[day] = shard["values"][:, 0, 0].astype(np.float64)
    stored = {name: values.astype(np.float32) for name, values in hourly.items()}
    for day in days:
        for channel, name in enumerate(FORCING):
            if name in FORCING[-3:]:
                window = (hours > day) & (hours <= day + pd.Timedelta(days=1))
                total = stored[name][window].astype(np.float64).sum()
                density = 1000.0 if name == "total_precipitation" else 1.0
                actual = shards[day][channel] * 86400.0 / density
            else:
                window = (hours >= day) & (hours < day + pd.Timedelta(days=1))
                total = stored[name][window].astype(np.float64).mean()
                actual = shards[day][channel]
            np.testing.assert_allclose(actual, total, rtol=1e-6, err_msg=name)


@given(
    st.lists(st.floats(0.0, 2500.0), min_size=2, max_size=40, unique=True),
    st.floats(-10.0, 10.0),
    st.floats(-0.01, 0.01),
    st.data(),
)
def test_depth_interpolation_is_exact_for_linear_profiles_and_never_bridges_gaps(
    depths, intercept, gradient, data
):
    """A profile linear in depth lands exactly on the OM4 centres it brackets.

    Centres above the shallowest or below the deepest observation stay missing,
    and a missing level spoils only the centres between its neighbours.
    """
    source = np.sort(np.asarray(depths))
    assume(np.diff(source).min() > 1e-3)
    profile = (intercept + gradient * source)[:, None, None]
    target = OM4_TARGET_DEPTHS
    inside = (target >= source[0]) & (target <= source[-1])
    result = interpolate_depth(profile, source, target)[:, 0, 0]
    np.testing.assert_array_equal(np.isfinite(result), inside)
    np.testing.assert_allclose(
        result[inside], intercept + gradient * target[inside], rtol=1e-5, atol=1e-4
    )
    k = data.draw(st.integers(0, len(source) - 1))
    gappy = profile.copy()
    gappy[k] = np.nan
    spoiled = inside.copy()
    if k > 0:
        spoiled &= target > source[k - 1]
    if k < len(source) - 1:
        spoiled &= target < source[k + 1]
    result = interpolate_depth(gappy, source, target)[:, 0, 0]
    np.testing.assert_array_equal(np.isnan(result), ~inside | spoiled)


def test_om4_nominal_layers_tile_the_column_the_stores_use():
    """Observation targets sit at the centres of the layers the published OM4 stores use."""
    assert (
        geometry.layer_thickness(OM4_LAYER_THICKNESS, OM4_INTERFACES, OM4_DEPTH_LEVELS)
        == []
    )


EPOCH = pd.Timestamp("1990-01-01")
CHANNELS = {"sst": 1, "adt": 3, "era5": len(FORCING)}


def linear_in_time(stamps) -> np.ndarray:
    """f(t) = days since EPOCH."""
    return np.asarray((pd.DatetimeIndex(stamps) - EPOCH) / pd.Timedelta(days=1))


class LinearArchive(ObservationArchive):
    """Daily shards that each hold the day's mean of f, i.e. f at the day's noon, in every channel.

    ``missing`` names one (product, day) whose shard is all NaN.
    """

    missing: tuple[str, pd.Timestamp] | None = None

    def day(self, product, date):
        stamp = pd.Timestamp(date)
        value = linear_in_time([stamp + pd.Timedelta(hours=12)])[0]
        if self.missing == (product, stamp):
            value = np.nan
        return np.full((CHANNELS[product], 1, 2), value)


@pytest.fixture(scope="module")
def archive_root(tmp_path_factory):
    """A grid and monthly IAP files; daily values come from `LinearArchive.day`."""
    root = tmp_path_factory.mktemp("archive")
    np.savez(
        root / "grid.npz",
        lat=[0.0],
        lon=[90.0, 270.0],
        mask=np.ones((77, 1, 2), dtype=bool),
    )
    for month in pd.period_range("1993-01", "2014-12", freq="M"):
        folder = root / "iap" / str(month.year)
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(
            folder / f"{month}.npz",
            values=np.zeros((2, OBSERVED_DEPTHS, 1, 2), dtype=np.float32),
            ohc=np.zeros((2, 1, 2), dtype=np.float32),
        )
    return root


months = st.dates(pd.Timestamp("1993-06-01").date(), pd.Timestamp("2012-07-01").date())


# An annual trajectory needs one complete month (7 steps for a 31-day month), and
# a replay of month 12 would run past 73 steps after a leap-year origin.
@given(months, st.integers(7, 73), st.integers(0, 10))
def test_five_day_means_are_stamped_at_their_midpoints(
    archive_root, start, steps, later
):
    """A field linear in time averages, over each five-day window, to its value at the stored midpoint.

    Holds for fresh samples, annual trajectories and replay months anchored at
    an earlier origin, in surface, velocity and atmosphere channels alike.
    """
    origin = pd.Timestamp(start).replace(day=1)
    archive = LinearArchive(archive_root)
    readers = [
        archive.sample(origin),
        trajectory(archive, origin, steps=steps),
        trajectory(archive, origin, target_month=origin + pd.DateOffset(months=later)),
    ]
    for sample in readers:
        expected = linear_in_time(sample["midpoints"])
        for key in ("surface", "velocity", "atmosphere"):
            values = sample[key].reshape(len(expected), -1)
            np.testing.assert_allclose(
                values, np.broadcast_to(expected[:, None], values.shape), atol=1e-6
            )


@given(months, st.sampled_from(sorted(CHANNELS)), st.data())
def test_a_missing_day_blanks_exactly_the_windows_that_contain_it(
    archive_root, start, product, data
):
    """No invented values: a NaN day removes its own product from every window it falls in, and nothing else."""
    origin = pd.Timestamp(start).replace(day=1)
    archive = LinearArchive(archive_root)
    windows = pd.DatetimeIndex(archive.sample(origin)["midpoints"]) - pd.Timedelta(
        days=STEP_DAYS / 2
    )
    gap = windows[data.draw(st.integers(0, len(windows) - 1))] + pd.Timedelta(
        days=data.draw(st.integers(0, STEP_DAYS - 1))
    )
    archive.missing = (product, gap)
    sample = archive.sample(origin)
    contains = (windows <= gap) & (gap < windows + pd.Timedelta(days=STEP_DAYS))
    layout = {
        "sst": ("surface", slice(0, 1)),
        "adt": ("surface", slice(1, 2)),
        "era5": ("atmosphere", slice(None)),
    }
    for name, (key, channels) in layout.items():
        missing = np.isnan(sample[key][:, channels]).all(axis=(1, 2, 3))
        finite = np.isfinite(sample[key][:, channels]).all(axis=(1, 2, 3))
        expected = contains if name == product else np.zeros_like(contains)
        np.testing.assert_array_equal(missing, expected, err_msg=name)
        np.testing.assert_array_equal(finite, ~expected, err_msg=name)
    velocity_missing = np.isnan(sample["velocity"]).all(axis=(1, 2, 3))
    np.testing.assert_array_equal(velocity_missing, contains & (product == "adt"))
