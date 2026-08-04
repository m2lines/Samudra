# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the spectral kernels.

Each pins a property against a signal whose spectrum is known analytically, so
a failure means the transform is wrong rather than that a reference number
drifted.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from samudra.metrics import spectra


def test_a_spatial_wave_lands_in_its_own_wavenumber_bin():
    """A single sinusoid must peak at its own wavelength, to within a bin.

    Also pins that the residual really is binning: halving the bin width must
    halve the offset. A transform error would not shrink that way.
    """
    size, extent = 128, 1000.0  # km
    spacing = extent / size
    wavelength = 125.0
    axis = np.arange(size) * spacing
    field = np.sin(2 * np.pi * np.add.outer(np.zeros(size), axis) / wavelength)

    errors = {}
    for bins in (32, 64, 128):
        wavenumber, power = spectra.isotropic_spectrum(
            field, dx=spacing, dy=spacing, num_bins=bins
        )
        peak = wavenumber[np.argmax(power)]
        half_bin = 0.5 * float(wavenumber[1] - wavenumber[0])
        errors[bins] = abs(peak - 1.0 / wavelength)
        assert errors[bins] <= 2 * half_bin

    # Refining the grid refines the answer, which is what distinguishes a
    # discretisation offset from a wrong transform.
    assert errors[64] < errors[32]
    assert errors[128] < errors[64]


def test_a_flat_field_has_no_spectrum_and_a_tiny_region_is_refused():
    """Constant input carries no power, and a region too small to transform fails."""
    wavenumber, power = spectra.isotropic_spectrum(
        np.full((64, 64), 3.0), dx=1.0, dy=1.0
    )
    assert np.allclose(power, 0.0, atol=1e-20)
    assert (wavenumber > 0).all()

    with pytest.raises(ValueError, match="too small"):
        spectra.isotropic_spectrum(np.zeros((4, 4)), dx=1.0, dy=1.0)


def test_welch_finds_an_annual_cycle_at_one_cycle_per_year():
    """A seasonal signal must peak at 1/year once the segments can resolve it."""
    time = pd.date_range("2015-01-03", "2022-12-29", freq="5D")
    series = pd.Series(
        np.sin(2 * np.pi * time.dayofyear.to_numpy() / 365.25), index=time
    )

    frequencies, power = spectra.welch_psd(series, nperseg=512)
    peak = frequencies[np.argmax(power)]
    assert peak == pytest.approx(1.0, abs=float(frequencies[1] - frequencies[0]))

    # Too short to segment at all: empty rather than a misleading spectrum.
    stub = series.iloc[:3]
    assert spectra.welch_psd(stub) == ((), ()) or all(
        arr.size == 0 for arr in spectra.welch_psd(stub)
    )


def test_log10_curve_rmse_measures_multiplicative_error():
    """Identical spectra score zero; a uniform factor of ten scores one dex."""
    wavenumber = np.logspace(-3, -1, 40)
    power = wavenumber**-2.0

    assert spectra.log10_rmse_between_curves(
        wavenumber, power, wavenumber, power
    ) == pytest.approx(0.0, abs=1e-12)
    assert spectra.log10_rmse_between_curves(
        wavenumber, power, wavenumber, power * 10
    ) == pytest.approx(1.0)

    # A factor of two counts the same wherever it occurs, which is the point of
    # working in logs: energetic and faint scales get equal weight.
    half = wavenumber.size // 2
    steep = power.copy()
    steep[half:] *= 2  # the small scales
    shallow = power.copy()
    shallow[:half] *= 2  # the energetic large scales
    assert spectra.log10_rmse_between_curves(
        wavenumber, power, wavenumber, steep
    ) == pytest.approx(
        spectra.log10_rmse_between_curves(wavenumber, power, wavenumber, shallow),
        rel=1e-6,
    )

    # Non-overlapping domains have no answer rather than a wrong one.
    assert np.isnan(
        spectra.log10_rmse_between_curves(wavenumber, power, wavenumber * 1e6, power)
    )


def test_interannual_band_averages_geometrically():
    """The band is a log mean, so it sits at the geometric centre and stays positive."""
    wavenumber = np.logspace(-3, -1, 30)
    power = wavenumber**-2.0
    curves = [(wavenumber, power * factor) for factor in (0.5, 1.0, 2.0)]

    axis, mean, lower, upper, years = spectra.interannual_band(curves)
    assert years == 3
    # An arithmetic mean of 0.5, 1 and 2 would be 1.167; the geometric one is 1.
    assert np.allclose(mean / power, 1.0)
    assert (lower > 0).all() and (lower < mean).all() and (mean < upper).all()
    assert axis.size == wavenumber.size

    empty = spectra.interannual_band([])
    assert empty[-1] == 0 and empty[0].size == 0


def test_region_spectrum_uses_physical_wavenumbers():
    """Regions are transformed in cycles per km, so latitudes stay comparable."""
    lat = np.linspace(25.0, 45.0, 64)
    lon = np.linspace(300.0, 320.0, 64)
    rng = np.random.default_rng(0)
    field = xr.DataArray(
        rng.normal(size=(lat.size, lon.size)),
        dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon},
    )

    wavenumber, power = spectra.region_spectrum(field, slice(300, 320), slice(25, 45))
    assert wavenumber.size and power.size
    # Degrees would put the first bin near 0.01; kilometres put it far lower.
    assert wavenumber.max() < 0.1
    assert np.isfinite(power).all()

    # A region smaller than the transform needs comes back empty, not raising.
    tiny, _ = spectra.region_spectrum(field, slice(300, 301), slice(25, 26))
    assert tiny.size == 0


def test_isotropic_spectrum_reproduces_the_reference_implementation():
    """Pin the transform against the reference suite it was ported from.

    The analytic tests above check that the spectrum is *a* correct spectrum;
    they pass under either Hann convention (symmetric or periodic) and either
    bin-edge convention, both of which shift power between neighbouring bins by
    a percent or more. Only fixed values from the reference pin those choices.

    Values come from `compute_isotropic_spectrum_torch` in the reference suite
    (`YuanYuan98/Ocean_Emulator:viz_jupyter/obs_evaluation.py` @ 33d4ae2e) as
    its spectral figures call it, on the field built below. The field is
    closed-form rather than random so the numbers do not depend on a generator
    implementation.

    That reference is outside this repository, so these numbers cannot be
    regenerated from a clean checkout. `tests/reference/regenerate_spectral_
    golden.py` reproduces them given a copy of that file, and records the exact
    call parameters used -- which are the reference's figure call sites, not its
    function defaults.

    The tolerance is loose because a bin edge and a mode wavenumber can be
    equal in exact arithmetic but differ in the last bit once computed, which
    moves a handful of modes between adjacent bins. On this grid that affects
    two bins of thirty-seven by about half a percent; everything else agrees to
    machine precision.
    """
    height, width = 120, 240
    dy = 0.125 * spectra.METRES_PER_DEGREE  # a 1/8 degree box, the DUACS grid
    dx = dy * np.cos(np.deg2rad(37.5))

    row, col = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    field = (
        np.sin(2 * np.pi * col / 40.0) * np.cos(2 * np.pi * row / 25.0)
        + 0.5 * np.sin(2 * np.pi * (col + 2 * row) / 13.0)
        + 0.25 * np.cos(2 * np.pi * col / 7.0)
        + 0.01 * col
        - 0.02 * row
        + 3.0
    )

    wavenumber, power = spectra.isotropic_spectrum(field, dx=dx, dy=dy)
    assert wavenumber.size == 37

    sampled = [0, 7, 15, 22, 28, 36]
    assert wavenumber[sampled] * spectra.RADIANS_PER_KM == pytest.approx(
        [
            3.027149616268e-03,
            4.540724424402e-02,
            9.384163810430e-02,
            1.362217327320e-01,
            1.725475281273e-01,
            2.209819219875e-01,
        ],
        rel=1e-10,
    )
    assert power[sampled] == pytest.approx(
        [
            1.365895408886e00,
            2.550728183756e-02,
            4.485457471581e00,
            1.767014781605e-04,
            9.309961775224e-06,
            9.342251409054e-07,
        ],
        rel=1e-3,
    )
