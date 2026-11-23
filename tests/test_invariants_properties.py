# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Each invariant accepts data built to satisfy it and flags a planted violation.

The data pipeline's property tests and the store audits trust these checks, so a
check that can't fail, or that fails on correct data, would hide bugs there.
"""

import cftime
import numpy as np
import pytest
import xarray as xr
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from samudra.invariants import (
    bounds,
    budgets,
    conservation,
    coupling,
    geometry,
    masks,
    releases,
    timeaxis,
)

pytestmark = pytest.mark.invariant(status="agreed")

finite = st.floats(-10.0, 10.0, allow_subnormal=False)


@st.composite
def nested_masks(draw):
    nlev, ny, nx = (
        draw(st.integers(2, 5)),
        draw(st.integers(2, 6)),
        draw(st.integers(2, 6)),
    )
    bottom = draw(arrays(np.int64, (ny, nx), elements=st.integers(0, nlev)))
    wet = np.arange(nlev)[:, None, None] < bottom[None]
    return xr.DataArray(wet, dims=("lev", "y", "x"))


@given(nested_masks(), st.data())
def test_mask_checks_accept_consistent_masks_and_flag_each_kind_of_disagreement(
    wet, data
):
    field = xr.where(wet, 1.0, np.nan)
    fraction = wet.astype(float) * 0.5
    assert masks.nest_with_depth(wet) == []
    assert masks.nan_matches_mask(field, wet, "f") == []
    assert masks.ocean_fraction_bounds(fraction) == []
    assert masks.mask_matches_fraction(wet, fraction, "m") == []

    lev = data.draw(st.integers(0, wet.sizes["lev"] - 1))
    y = data.draw(st.integers(0, wet.sizes["y"] - 1))
    x = data.draw(st.integers(0, wet.sizes["x"] - 1))
    flipped = wet.copy()
    flipped.values[lev, y, x] = not wet.values[lev, y, x]
    assert masks.nan_matches_mask(field, flipped, "f") != []
    assert masks.mask_matches_fraction(flipped, fraction, "m") != []
    if lev > 0 and not wet.values[lev - 1, y, x]:
        assert masks.nest_with_depth(flipped) != []
    bigger = fraction.copy()
    bigger.values[lev, y, x] = 1.5
    assert masks.ocean_fraction_bounds(bigger) != []


@given(st.floats(0.0, 40.0), st.floats(0.0, 6000.0), st.floats(0.11, 5.0))
def test_freezing_check_flags_only_water_below_its_freezing_point(
    salinity, depth, below
):
    point = bounds.freezing_point(salinity, depth)
    temperature = xr.DataArray([point, point + 1.0])
    so = xr.DataArray([salinity, salinity])
    assert bounds.above_freezing(temperature, so, depth) == []
    assert bounds.above_freezing(temperature - below, so, depth) != []


@given(
    st.integers(4, 8), st.integers(4, 8), st.floats(30.0, 36.0), st.floats(0.0, 0.45)
)
def test_coastal_dilution_flags_a_coast_dragged_towards_land(
    ny, nx, salinity, fraction
):
    wet = np.ones((ny, nx), bool)
    wet[0, :] = False  # a southern coastline
    field = xr.DataArray(np.where(wet, salinity, np.nan), dims=("y", "x"))
    wet_da = xr.DataArray(wet, dims=("y", "x"))
    assert bounds.coastal_dilution(field, wet_da, "so") == []
    diluted = field.copy()
    diluted[1, 0] = fraction * salinity
    assert bounds.coastal_dilution(diluted, wet_da, "so") != []


@given(st.integers(2, 30), st.integers(3, 60), st.floats(6300e3, 6400e3))
def test_sphere_area_accepts_the_right_radius_only(ny, nx, radius):
    lat_b = np.repeat(np.linspace(-90, 90, ny + 1)[:, None], nx + 1, axis=1)
    lon_b = np.repeat(np.linspace(0, 360, nx + 1)[None, :], ny + 1, axis=0)
    area = xr.DataArray(
        geometry.rectilinear_cell_area(lat_b, lon_b, radius), dims=("y", "x")
    )
    assert geometry.sphere_area(area, radius=radius, rtol=1e-9) == []
    assume(abs(radius - geometry.EARTH_RADIUS_M) > 1e3)
    assert geometry.sphere_area(area, rtol=1e-4) != []


@given(st.integers(1, 19), st.data())
def test_layer_thickness_flags_any_inconsistent_layer(nlev, data):
    ilev = np.array(
        [
            0,
            5,
            15,
            30,
            50,
            80,
            130,
            200,
            300,
            450,
            650,
            900,
            1200,
            1600,
            2100,
            2700,
            3500,
            4500,
            5500,
            6500,
        ],
        float,
    )[: nlev + 1]
    dz = np.diff(ilev)
    lev = 0.5 * (ilev[1:] + ilev[:-1])
    assert geometry.layer_thickness(dz, ilev, lev) == []
    k = data.draw(st.integers(0, nlev - 1))
    broken = dz.copy()
    broken[k] += data.draw(st.floats(0.5, 100.0))
    assert geometry.layer_thickness(broken, ilev, lev) != []


@st.composite
def om4_axes(draw):
    calendar = draw(st.sampled_from(["julian", "noleap", "proleptic_gregorian"]))
    year = draw(st.integers(1958, 2018))
    stamps, windows = [], []
    for y in range(year, year + draw(st.integers(1, 4))):
        start = cftime.datetime(y, 1, 1, calendar=calendar)
        for i in range(73):
            begin = start + i * np.timedelta64(5, "D").astype(object)
            windows.append((begin, begin + np.timedelta64(5, "D").astype(object)))
            stamps.append(begin + np.timedelta64(60, "h").astype(object))
    return np.array(stamps), np.array(windows)


@given(om4_axes(), st.data())
def test_time_checks_accept_the_om4_axis_and_flag_a_moved_stamp(axis, data):
    stamps, windows = axis
    assert timeaxis.om4_five_day_axis(stamps) == []
    assert timeaxis.averaging_windows(stamps, windows, leap_day_gaps=True) == []
    i = data.draw(st.integers(0, len(stamps) - 1))
    hours = data.draw(st.sampled_from([-24, -12, 12, 24]))
    moved = stamps.copy()
    moved[i] = moved[i] + np.timedelta64(hours, "h").astype(object)
    assert timeaxis.averaging_windows(moved, windows, leap_day_gaps=True) != []
    assert timeaxis.om4_five_day_axis(moved) != []


@given(st.integers(2, 20), st.integers(0, 2**32 - 1), st.integers(1, 4))
def test_window_means_accept_true_means_and_flag_shifted_ones(windows, seed, shift):
    days = 5 * windows
    start = cftime.datetime(2001, 1, 1, calendar="noleap")
    day = np.timedelta64(1, "D").astype(object)
    daily_bounds = np.array(
        [(start + i * day, start + (i + 1) * day) for i in range(days)]
    )
    five_bounds = np.stack([daily_bounds[::5, 0], daily_bounds[4::5, 1]], axis=1)
    values = np.random.default_rng(seed).normal(size=days)
    daily = xr.DataArray(values, dims="time")
    means = xr.DataArray(values.reshape(windows, 5).mean(axis=1), dims="time")
    assert timeaxis.window_mean_preserved(daily, daily_bounds, means, five_bounds) == []
    rolled = xr.DataArray(
        np.roll(values, shift).reshape(windows, 5).mean(axis=1), dims="time"
    )
    assume(not np.allclose(rolled, means, rtol=1e-5, atol=1e-6))
    assert (
        timeaxis.window_mean_preserved(daily, daily_bounds, rolled, five_bounds) != []
    )
    # On an all-zero field no value can differ, so only coverage and NaN handling
    # flag these: a repeated fine window counts its days twice, and a NaN day has
    # no value to average.
    zeros, zero_means = xr.zeros_like(daily), xr.zeros_like(means)
    repeated = xr.concat([zeros, zeros.isel(time=[0])], dim="time")
    repeated_bounds = np.concatenate([daily_bounds, daily_bounds[:1]])
    assert (
        timeaxis.window_mean_preserved(
            repeated, repeated_bounds, zero_means, five_bounds
        )
        != []
    )
    gap = zeros.copy()
    gap[seed % days] = np.nan
    assert (
        timeaxis.window_mean_preserved(gap, daily_bounds, zero_means, five_bounds) != []
    )


@given(om4_axes(), st.integers(-48, 48).filter(bool))
def test_windows_match_compares_datetime64_with_cftime(axis, hours):
    """xarray decodes standard calendars to datetime64, so either side may be either kind."""
    _, windows = axis
    gregorian = np.array(
        [
            [
                cftime.datetime(*t.timetuple()[:6], calendar="proleptic_gregorian")
                for t in w
            ]
            for w in windows
        ]
    )
    as_datetime64 = np.array(
        [[np.datetime64(t.isoformat(), "ns") for t in w] for w in gregorian]
    )
    assert timeaxis.windows_match(as_datetime64, gregorian) == []
    shifted = as_datetime64 + np.timedelta64(hours, "h")
    assert timeaxis.windows_match(shifted, gregorian) != []


@given(
    arrays(np.float64, (6, 8), elements=finite),
    arrays(np.float64, (6, 8), elements=st.floats(0.1, 10.0)),
    st.floats(1.01, 2.0),
)
def test_conservation_accepts_block_means_and_flags_a_scaled_copy(
    field, weights, scale
):
    fine = xr.DataArray(field, dims=("y", "x"))
    w = xr.DataArray(weights, dims=("y", "x"))
    w_coarse = w.coarsen(y=2, x=2).reduce(np.sum)
    coarse = (fine * w).coarsen(y=2, x=2).reduce(np.sum) / w_coarse
    found = (
        conservation.integral_preserved(fine, w, coarse, w_coarse, "f", rtol=1e-9)
        + conservation.mean_preserved(
            fine, w, coarse, w_coarse, "f", rtol=1e-9, atol=1e-12
        )
        + conservation.variance_not_increased(fine, w, coarse, w_coarse, "f")
        + conservation.no_new_extremes(fine, coarse, "f", atol=1e-12)
    )
    assert found == []
    # Scaling by 1.01 is visible at rtol=1e-3 only if the total isn't mostly
    # cancellation: rtol is relative to the integral of |field|.
    # Nothing to integrate is a failure, not 0 == 0.
    land = xr.full_like(fine, np.nan)
    assert conservation.integral_preserved(land, w, land, w, "f") != []
    assume(float(abs((fine * w).sum())) > 0.2 * float((abs(fine) * w).sum()))
    assert (
        conservation.integral_preserved(
            fine, w, coarse * scale, w_coarse, "f", rtol=1e-3
        )
        != []
    )


@given(st.integers(30, 120), st.integers(0, 2**32 - 1), st.sampled_from([-1, 1]))
def test_lag_check_finds_a_forcing_one_step_off(n, seed, shift):
    rng = np.random.default_rng(seed)
    forcing = np.cumsum(rng.normal(size=n)) + rng.normal(size=n)
    response = forcing + 0.1 * rng.normal(size=n)
    assert coupling.response_peaks_at_lag_zero(forcing, response, "f") == []
    shifted = np.roll(forcing, shift)
    shifted[0 if shift == 1 else -1] = np.nan
    assert coupling.response_peaks_at_lag_zero(shifted, response, "f") != []


@given(st.integers(20, 100), st.integers(0, 2**32 - 1))
def test_heat_budget_accepts_heat_that_enters_at_the_surface(n, seed):
    rng = np.random.default_rng(seed)
    step = 5 * 86400.0
    t = np.arange(n)
    flux = 1e15 * np.sin(2 * np.pi * t / 73) + 1e14 * rng.normal(size=n)
    # Window-mean heat content of an ocean warmed only by `flux`.
    content = 1e25 + np.concatenate(
        [[0.0], np.cumsum(0.5 * (flux[1:] + flux[:-1]) * step)]
    )
    seconds = t * step
    assert (
        budgets.heat_budget(
            content, flux, seconds, min_correlation=0.8, slope_range=(0.7, 1.3)
        )
        == []
    )
    assert budgets.heat_budget(content, -flux, seconds) != []


@given(arrays(np.float32, (3, 4), elements=finite), st.data())
def test_release_comparison_flags_any_changed_value(values, data):
    a = xr.Dataset({"thetao": (("y", "x"), values)})
    assert releases.stores_agree(a, a.copy(deep=True)) == []
    y, x = data.draw(st.integers(0, 2)), data.draw(st.integers(0, 3))
    changed = a.copy(deep=True)
    changed.thetao[y, x] = values[y, x] + 1.0
    assert releases.stores_agree(a, changed) != []
    assert (
        releases.stores_agree(a, changed, expected_changes=frozenset({"thetao"})) == []
    )


def cell_means(edges_deg: np.ndarray, lon_edges: np.ndarray, a, b, k, phase):
    """Exact cell means of (a + b sin(lat)) * (1 + cos(k lon + phase)) on a lat-lon grid."""
    s = np.sin(np.deg2rad(edges_deg))
    row = a + b * (s[:-1] + s[1:]) / 2  # mean of a + b sin(lat) over d(sin lat)
    x = np.deg2rad(lon_edges)
    col = 1 + (np.sin(k * x[1:] + phase) - np.sin(k * x[:-1] + phase)) / (
        k * np.diff(x)
    )
    return xr.DataArray(np.outer(row, col), dims=("y", "x"))


@given(
    st.integers(8, 20),
    st.integers(2, 6),
    st.floats(0.05, 0.95),
    st.integers(1, 4),
    st.floats(0, 2 * np.pi),
)
def test_zonal_spectra_accept_non_nested_rows_and_flag_rescaled_waves(
    fine_rows, coarse_rows, offset, k, phase
):
    # Rows that don't share edges.
    fine_edges = np.linspace(-63.0, -54.0, fine_rows + 1)
    coarse_step = 6.0 / (coarse_rows + 1)
    coarse_edges = -62.0 + (offset + np.arange(coarse_rows + 1)) * coarse_step
    fine_lon = np.linspace(0.0, 360.0, 129)
    coarse_lon = fine_lon[::4]
    args = (30.0, 5.0, k, phase)
    fine = cell_means(fine_edges, fine_lon, *args)
    coarse = cell_means(coarse_edges, coarse_lon, *args)
    # A cell holds its whole row's mean, so only a band on the coarse edges is
    # exact; the fine grid's partial rows then cost well under 1e-3.
    band = (float(coarse_edges[0]), float(coarse_edges[-1]))
    assert (
        conservation.zonal_spectra_agree(
            fine, fine_edges, coarse, coarse_edges, band, "f", rtol=1e-3
        )
        == []
    )
    wave = coarse - coarse.mean("x")
    assert (
        conservation.zonal_spectra_agree(
            fine, fine_edges, coarse + 0.2 * wave, coarse_edges, band, "f"
        )
        != []
    )


def test_rectilinear_cell_area_refuses_a_curvilinear_grid():
    lon_b, lat_b = np.meshgrid(
        np.linspace(0.0, 4.0, 5), np.array([-60.0, -58.0, -56.0])
    )
    with pytest.raises(ValueError, match="rectilinear"):
        geometry.rectilinear_cell_area(lat_b + np.linspace(0.0, 1.0, 5), lon_b)


def test_band_spectrum_refuses_curvilinear_rows_and_extra_dimensions():
    field = xr.DataArray(np.ones((2, 4)), dims=("y", "x"))
    edges = np.repeat(np.array([[-60.0], [-58.0], [-56.0]]), 5, axis=1)
    tilted = edges + np.linspace(0.0, 1.0, 5)
    with pytest.raises(ValueError, match="rectilinear"):
        conservation.band_zonal_spectrum(field, tilted, (-60.0, -56.0))
    with pytest.raises(ValueError, match="dimensions"):
        conservation.band_zonal_spectrum(
            field.expand_dims(time=2), edges, (-60.0, -56.0)
        )
