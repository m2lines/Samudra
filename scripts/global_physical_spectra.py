# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Regional diagnostic pseudo-spectra on common wet support, including broad boxes."""

import numpy as np

REGIONS = [
    ("Gulf Stream", (300, 320), (25, 45)),
    ("Kuroshio", (150, 170), (25, 45)),
    ("Agulhas", (40, 60), (-50, -30)),
    ("Malvinas", (311, 331), (-51, -31)),
    ("Niño 3.4", (190, 240), (-5, 5)),
    ("Tropical Pacific", (130, 290), (-30, 30)),
]


def spectrum(field, lat, lon, support):
    z = np.asarray(field, dtype=float)
    valid = np.asarray(support, bool) & np.isfinite(z)
    h, w = z.shape
    if min(h, w) < 8 or valid.sum() < h * w * 0.3:
        return None
    yy, xx = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij")
    design = np.stack([xx, yy, np.ones_like(xx)], axis=-1)
    coefficients = np.linalg.lstsq(design[valid], z[valid], rcond=None)[0]
    residual = np.where(valid, z - design @ coefficients, 0)
    window = np.outer(
        0.5 - 0.5 * np.cos(2 * np.pi * np.arange(h) / h),
        0.5 - 0.5 * np.cos(2 * np.pi * np.arange(w) / w),
    )
    correction = np.mean(window**2 * valid)
    dy = float(np.median(np.diff(lat))) * 111.32
    dx = float(np.median(np.diff(lon))) * 111.32 * np.cos(np.deg2rad(np.mean(lat)))
    assert dx > 0 and dy > 0
    power = (
        np.abs(np.fft.rfft2(window * residual, norm="forward")) ** 2
        / correction
        * (w * dx)
        * (h * dy)
    )
    ky, kx = np.meshgrid(
        np.fft.fftfreq(h, d=dy), np.fft.rfftfreq(w, d=dx), indexing="ij"
    )
    k = np.hypot(kx, ky)
    nyquist = min(1 / (2 * dx), 1 / (2 * dy))
    nbins = max(3, min(h, w) // 4)
    edges = np.linspace(0, nyquist, nbins + 1)
    center = 0.5 * (edges[1:] + edges[:-1])
    idx = np.searchsorted(edges, k, side="right") - 1
    keep = (idx >= 0) & (idx < nbins)
    totals = np.bincount(idx[keep], weights=power[keep], minlength=nbins)
    counts = np.bincount(idx[keep], minlength=nbins)
    # Convert cycle-frequency density to angular-wavenumber density:
    # S_k = S_f / (2π)^2, hence k S_k = f S_f / (2π).
    p = (
        np.divide(totals, counts, out=np.full(nbins, np.nan), where=counts > 0)
        * center
        / (2 * np.pi)
    )
    return {
        "k_rad_km": (center * 2 * np.pi).tolist(),
        "power": p.tolist(),
        "finite_fraction": float(valid.mean()),
        "shape": [h, w],
        "nyquist_rad_km": float(2 * np.pi * nyquist),
    }


def geostrophic(ssh, lat, lon):
    z = np.asarray(ssh)
    phi = np.deg2rad(lat)
    f = 2 * 7.292115e-5 * np.sin(phi)
    dy = np.gradient(z, np.deg2rad(lat), axis=-2) / 6371000
    dx = np.gradient(z, np.deg2rad(lon), axis=-1) / (6371000 * np.cos(phi)[:, None])
    safe = np.where(np.abs(lat) >= 5, f, np.nan)[:, None]
    return -9.81 * dy / safe, 9.81 * dx / safe
