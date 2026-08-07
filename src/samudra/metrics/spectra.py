# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Spectral kernels for observation-based evaluation.

Spectra answer a question the RMSE metrics cannot: *at which scales* a model
gets the ocean wrong. A rollout can match the observed variance overall and
still put it at the wrong wavelengths, and a model that smooths eddies away
looks fine on a map average while its spectrum falls off early.

Like `kernels`, everything here is a pure function over arrays. Errors are
reported in log10 space (dex), because power spectra span many orders of
magnitude and a linear residual would be dominated entirely by the largest
scales.

Implements the spectral metrics of the metrics document [1].

[1]: https://docs.google.com/document/d/1cWzIrcajfW-gvoMzV3PU0JDAouVt5xerRmEiBIUbV1s
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xarray as xr
from scipy import signal

logger = logging.getLogger(__name__)

# Wavenumber bins per spatial spectrum, as a fraction of the shorter grid side.
DEFAULT_BIN_FACTOR = 2

# Smallest region side a spatial spectrum is computed for.
MIN_REGION_SIDE = 8

METRES_PER_DEGREE = 111_320.0

# Cycles per metre to radians per kilometre.
RADIANS_PER_KM = 2.0 * np.pi * 1000.0

# Welch segment length. Long enough to resolve the seasonal cycle at 5-day
# sampling, short enough to leave several independent segments in an 8-year
# record.
DEFAULT_NPERSEG = 256

# Regions the spectra are reported over: open-ocean boxes away from coasts,
# plus the two western-boundary systems where mesoscale activity is strongest.
SPATIAL_REGIONS: tuple[tuple[str, slice, slice], ...] = (
    ("North Pacific", slice(170, 200), slice(25, 45)),
    ("Gulf Stream", slice(300, 320), slice(25, 45)),
    ("Agulhas", slice(40, 60), slice(-50, -30)),
)
TEMPORAL_REGIONS: tuple[tuple[str, slice, slice], ...] = (
    ("Global", slice(0, 360), slice(-90, 90)),
    ("Gulf Stream", slice(300, 320), slice(25, 45)),
    ("Agulhas", slice(40, 60), slice(-50, -30)),
)


def _periodic_hann(size: int) -> np.ndarray:
    """Hann window with no repeated endpoint (scipy's "sym=False")."""
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(size) / size)


def remove_plane(field: np.ndarray) -> np.ndarray:
    """Subtract a least-squares 2-D plane from the last two axes.

    A spatial trend across the box is a boundary effect, not a physical scale.
    Left in place it leaks power across every wavenumber once the window is
    applied.
    """
    *lead, height, width = field.shape
    y = np.linspace(-1.0, 1.0, height)
    x = np.linspace(-1.0, 1.0, width)
    grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
    design = np.stack([grid_x.ravel(), grid_y.ravel(), np.ones(grid_x.size)], axis=1)
    flat = field.reshape(-1, height * width)
    coefficients, *_ = np.linalg.lstsq(design, flat.T, rcond=None)
    plane = (design @ coefficients).T.reshape(-1, height, width)
    return (flat.reshape(-1, height, width) - plane).reshape(*lead, height, width)


def isotropic_spectrum(
    field: np.ndarray,
    dx: float = 1.0,
    dy: float = 1.0,
    num_bins: int | None = None,
    detrend: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Radially averaged power spectrum of a 2-D field, or a stack of them.

    Args:
        field: `(height, width)`, or `(n, height, width)` for a stack.
        dx: Grid spacing along the last axis, in the units the wavenumber
            should be reciprocal to.
        dy: Grid spacing along the second-to-last axis.
        num_bins: Wavenumber bins across the full Fourier plane, before the
            bins above the lower Nyquist are dropped; defaults to the shorter
            side over `DEFAULT_BIN_FACTOR`. Fewer bins come back than are asked
            for, which is why the returned wavenumbers must be used rather than
            assumed.
        detrend: Remove a 2-D plane before windowing.

    Returns:
        Bin-center wavenumbers and the isotropic spectrum, in cycles per unit
        of `dx`, the latter shaped `(bins,)` or `(n, bins)` to match the input.

    The spectrum is multiplied by its bin-center wavenumber, so a flat line on
    log-log axes means equal energy per octave and the area under the curve is
    the variance.
    """
    array = np.asarray(field, dtype=float)
    stacked = array.ndim == 3
    if not stacked:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError(f"Expected a 2-D field or a stack of them, got {array.shape}")

    n, height, width = array.shape
    if num_bins is None:
        num_bins = min(height, width) // DEFAULT_BIN_FACTOR
    if num_bins < 2:
        raise ValueError(
            f"A {height}x{width} region is too small for a spectrum; it yields "
            f"{num_bins} wavenumber bin(s), and a single bin describes no slope"
        )

    array = (
        remove_plane(array)
        if detrend
        else array - array.mean(axis=(-2, -1), keepdims=True)
    )

    # A Hann window in both directions suppresses the discontinuity at the box
    # edges, which would otherwise ring across all wavenumbers. Dividing by the
    # mean squared window restores the variance the taper removes.
    # Periodic ("DFT-even") Hann, not the symmetric np.hanning: an FFT treats
    # the box as one period, so the window must not repeat its endpoint.
    window = np.outer(_periodic_hann(height), _periodic_hann(width))
    correction = float(np.mean(window**2))
    power = np.abs(np.fft.rfft2(array * window, norm="forward")) ** 2
    spectrum_2d = power / correction * (width * dx) * (height * dy)

    k_x: np.ndarray = np.fft.rfftfreq(width, d=dx)
    k_y: np.ndarray = np.fft.fftfreq(height, d=dy)
    grid_y, grid_x = np.meshgrid(k_y, k_x, indexing="ij")
    # sqrt of the sum of squares rather than np.hypot: on a regular grid many
    # modes fall exactly on a bin edge, and the two formulas disagree in the
    # last bit, which moves those modes between neighbouring bins.
    k_magnitude = np.sqrt(grid_x**2 + grid_y**2)

    # Bins span the whole Fourier plane out to its corner, and those beyond the
    # lower Nyquist are then dropped. Sizing the bins to the corner rather than
    # to the Nyquist is what keeps bin width independent of how much of the
    # plane survives truncation. Nothing above the lower Nyquist is retained:
    # past it the two axes disagree about what is resolved.
    nyquist = min(1.0 / (2.0 * dx), 1.0 / (2.0 * dy))
    edges = np.linspace(0.0, float(k_magnitude.max()), num_bins + 1)
    edges = edges[edges < nyquist]
    num_bins = edges.size - 1
    if num_bins < 2:
        raise ValueError(
            f"A {height}x{width} region is too small for a spectrum; it yields "
            f"{max(num_bins, 0)} wavenumber bin(s) below the Nyquist, and a "
            f"single bin describes no slope"
        )
    centers = 0.5 * (edges[:-1] + edges[1:])

    flat_k = k_magnitude.ravel()
    retained = flat_k <= edges[-1]
    # right=False here: a mode landing exactly on a bin edge belongs to the
    # bin above it. The opposite convention shifts every such mode down one
    # bin, and on a regular grid exact edge hits are common rather than rare.
    index = np.digitize(flat_k[retained], edges[1:-1], right=False)
    flat_psd = spectrum_2d.reshape(n, -1)[:, retained]

    totals = np.zeros((n, num_bins))
    for row, values in zip(totals, flat_psd, strict=True):
        np.add.at(row, index, values)
    counts = np.bincount(index, minlength=num_bins).astype(float)
    # A bin no mode fell into has no answer, not an answer of zero: zero would
    # drop silently off a log axis and out of a log-space curve comparison,
    # making "nothing sampled here" indistinguishable from "no energy here".
    with np.errstate(invalid="ignore"):
        isotropic = np.where(counts > 0, totals / np.maximum(counts, 1.0), np.nan)
    isotropic = isotropic * centers

    return centers, isotropic if stacked else isotropic[0]


def _open_ocean_block(field: xr.DataArray, dims: tuple[str, str]) -> np.ndarray | None:
    """Return the region's values, or None if any cell is missing.

    A 2-D FFT assumes a complete rectangle. Filling land or missing cells puts
    step changes in the interior of the box, which a window cannot taper away
    and which leak power across every wavenumber -- so the spectral regions are
    curated open-ocean boxes and a box that is not fully observed gets no
    spectrum at all.
    """
    values = np.asarray(field.transpose(*dims).values, dtype=float)
    return values if np.isfinite(values).all() else None


def region_spectrum(
    field: xr.DataArray,
    lon_range: slice,
    lat_range: slice,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
    name: str = "region",
) -> tuple[np.ndarray, np.ndarray]:
    """Isotropic spectrum of a region, averaged over time if the field has it.

    Returns wavenumbers in radians per kilometre, so a curve reads directly as
    wavelengths (2*pi/k) and regions at different latitudes stay comparable.
    Returns empty arrays, and logs why, for a region that is too small or not
    fully observed -- the caller draws the panel as unavailable rather than
    showing a spectrum built from filled-in land.
    """
    region = field.sel({lon_dim: lon_range, lat_dim: lat_range})
    if (
        region.sizes[lat_dim] < MIN_REGION_SIDE
        or region.sizes[lon_dim] < MIN_REGION_SIDE
    ):
        logger.info(
            "no spectrum for %s: box is %dx%d, below the %dx%d minimum",
            name,
            region.sizes[lat_dim],
            region.sizes[lon_dim],
            MIN_REGION_SIDE,
            MIN_REGION_SIDE,
        )
        return np.array([]), np.array([])

    # Degrees to metres at the region's mid-latitude. One factor for the whole
    # box keeps the transform on a regular grid; the boxes are small enough
    # that the cosine varies little across them.
    lat = np.asarray(region[lat_dim].values, dtype=float)
    lon = np.asarray(region[lon_dim].values, dtype=float)
    dy = abs(float(np.median(np.diff(lat)))) * METRES_PER_DEGREE
    dx = (
        abs(float(np.median(np.diff(lon))))
        * METRES_PER_DEGREE
        * float(np.cos(np.deg2rad(lat.mean())))
    )
    if not (dx > 0 and dy > 0):
        logger.info("no spectrum for %s: grid spacing is not positive", name)
        return np.array([]), np.array([])

    if "time" in region.dims:
        # Every step or none: averaging over whichever steps happened to be
        # fully observed would weight the result towards those conditions.
        stack = []
        unobserved = []
        for step in range(region.sizes["time"]):
            block = _open_ocean_block(region.isel(time=step), (lat_dim, lon_dim))
            if block is None:
                unobserved.append(step)
                continue
            stack.append(block)
        if unobserved:
            logger.info(
                "no spectrum for %s: %d of %d steps have land or missing cells "
                "(first at index %d)",
                name,
                len(unobserved),
                region.sizes["time"],
                unobserved[0],
            )
            return np.array([]), np.array([])
        if not stack:
            logger.info("no spectrum for %s: no time steps", name)
            return np.array([]), np.array([])
        try:
            centers, spectra = isotropic_spectrum(np.stack(stack), dx=dx, dy=dy)
        except ValueError as error:
            logger.info("no spectrum for %s: %s", name, error)
            return np.array([]), np.array([])
        return centers * RADIANS_PER_KM, spectra.mean(axis=0)

    block = _open_ocean_block(region, (lat_dim, lon_dim))
    if block is None:
        logger.info("no spectrum for %s: box contains land or missing cells", name)
        return np.array([]), np.array([])
    try:
        centers, spectrum = isotropic_spectrum(block, dx=dx, dy=dy)
    except ValueError as error:
        # A box can pass the side-length check and still leave too few bins
        # below the lower Nyquist once the spacings are strongly anisotropic.
        # The caller draws that as unavailable, like any other region it cannot
        # transform, rather than losing the whole figure.
        logger.info("no spectrum for %s: %s", name, error)
        return np.array([]), np.array([])
    return centers * RADIANS_PER_KM, spectrum


def region_mean_series(
    field: xr.DataArray,
    lon_range: slice,
    lat_range: slice,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
    max_points: int = 100_000,
) -> pd.Series:
    """Cosine-latitude-weighted mean of a region, as a time series.

    Large global grids are regularly subsampled first: a temporal spectrum
    needs the time axis intact, not every cell, and the weighted mean converges
    long before the full grid is used.
    """
    region = field.sel({lon_dim: lon_range, lat_dim: lat_range})
    if region.sizes[lat_dim] * region.sizes[lon_dim] > max_points:
        region = region.isel(
            {
                lat_dim: slice(None, None, max(1, region.sizes[lat_dim] // 180)),
                lon_dim: slice(None, None, max(1, region.sizes[lon_dim] // 360)),
            }
        )

    weights = np.cos(np.deg2rad(region[lat_dim]))
    weighted = region.weighted(weights.fillna(0.0))
    series = weighted.mean((lat_dim, lon_dim), skipna=True)
    return pd.Series(
        np.asarray(series.values, dtype=float),
        index=pd.DatetimeIndex(series["time"].values),
    )


def welch_psd(
    series: pd.Series, nperseg: int = DEFAULT_NPERSEG
) -> tuple[np.ndarray, np.ndarray]:
    """Welch power spectral density of a scalar series, in cycles per year.

    Returns empty arrays when the series is too short to segment. The zero
    frequency is dropped: it carries the mean, not a timescale, and cannot be
    plotted on log axes.
    """
    values = series.astype(float).dropna()
    if values.size < 4:
        return np.array([]), np.array([])

    index = pd.DatetimeIndex(values.index)
    steps = np.diff(index.values).astype("timedelta64[s]").astype(float) / 86400.0
    steps = steps[np.isfinite(steps) & (steps > 0)]
    if steps.size == 0:
        return np.array([]), np.array([])

    frequencies, power = signal.welch(
        values.to_numpy(),
        fs=365.25 / float(np.median(steps)),
        detrend="linear",
        window="hann",
        scaling="density",
        nperseg=min(nperseg, values.size),
    )
    positive = frequencies > 0
    return frequencies[positive], power[positive]


def log10_rmse_between_curves(
    x_reference: np.ndarray,
    y_reference: np.ndarray,
    x_other: np.ndarray,
    y_other: np.ndarray,
) -> float:
    """RMS difference between two spectra in log10 space, in dex.

    The comparison curve is interpolated onto the reference wavenumbers in
    log-log space, over the domain both cover. Working in logs measures
    multiplicative error, so a factor-of-two miss counts the same whether it
    happens at the energetic large scales or the faint small ones -- in linear
    units the largest bin would decide the whole score.
    """
    x_reference = np.asarray(x_reference, dtype=float)
    y_reference = np.asarray(y_reference, dtype=float)
    x_other = np.asarray(x_other, dtype=float)
    y_other = np.asarray(y_other, dtype=float)

    reference_ok = (
        np.isfinite(x_reference)
        & np.isfinite(y_reference)
        & (x_reference > 0)
        & (y_reference > 0)
    )
    other_ok = (
        np.isfinite(x_other) & np.isfinite(y_other) & (x_other > 0) & (y_other > 0)
    )
    if not reference_ok.any() or other_ok.sum() < 2:
        return float("nan")

    order = np.argsort(x_other[other_ok])
    x_sorted = x_other[other_ok][order]
    y_sorted = y_other[other_ok][order]
    x_sorted, unique = np.unique(x_sorted, return_index=True)
    y_sorted = y_sorted[unique]
    if x_sorted.size < 2:
        return float("nan")

    domain = reference_ok & (x_reference >= x_sorted[0]) & (x_reference <= x_sorted[-1])
    if not domain.any():
        return float("nan")

    interpolated = np.interp(
        np.log10(x_reference[domain]), np.log10(x_sorted), np.log10(y_sorted)
    )
    error = np.log10(y_reference[domain]) - interpolated
    error = error[np.isfinite(error)]
    if error.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(error**2)))


def interannual_band(
    curves: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Geometric mean and +/- 1 log-standard-deviation band across yearly spectra.

    Averaging is done on logs because spectra span orders of magnitude: an
    arithmetic mean would be pinned to the most energetic year, and a linear
    standard deviation would put the lower band below zero, which cannot be
    drawn on log axes.

    Returns `(x, mean, lower, upper, n_years)`, all arrays empty when no curve
    is usable.
    """
    cleaned = []
    for x, y in curves:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        usable = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        if usable.sum() >= 2:
            order = np.argsort(x[usable])
            cleaned.append((x[usable][order], y[usable][order]))

    empty = np.array([])
    if not cleaned:
        return empty, empty, empty, empty, 0

    x_reference = max((x for x, _ in cleaned), key=lambda arr: arr.size)
    rows = np.asarray(
        [
            np.interp(
                np.log(x_reference), np.log(x), np.log(y), left=np.nan, right=np.nan
            )
            for x, y in cleaned
        ]
    )
    with np.errstate(invalid="ignore"):
        log_mean = np.nanmean(rows, axis=0)
        log_std = (
            np.nanstd(rows, axis=0, ddof=1)
            if rows.shape[0] > 1
            else np.zeros_like(log_mean)
        )
    return (
        x_reference,
        np.exp(log_mean),
        np.exp(log_mean - log_std),
        np.exp(log_mean + log_std),
        rows.shape[0],
    )
