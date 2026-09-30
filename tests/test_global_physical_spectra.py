# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Numerical checks for the separate broad-region diagnostic transform."""

import importlib.util
from pathlib import Path

import numpy as np

path = Path(__file__).parents[1] / "scripts" / "global_physical_spectra.py"
spec = importlib.util.spec_from_file_location("global_physical_spectra", path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_masked_spectrum_removes_plane_and_scales_power():
    lat = np.linspace(25, 45, 40)
    lon = np.linspace(150, 170, 40)
    rng = np.random.default_rng(5)
    field = rng.normal(size=(40, 40))
    support = np.ones(field.shape, bool)
    support[:3, :7] = False
    y, x = np.meshgrid(np.linspace(-1, 1, 40), np.linspace(-1, 1, 40), indexing="ij")
    a = module.spectrum(field, lat, lon, support)
    b = module.spectrum(2 * field + 3 * x - 7 * y + 37, lat, lon, support)
    np.testing.assert_allclose(b["power"], 4 * np.array(a["power"]), rtol=1e-10)
    assert max(a["k_rad_km"]) < a["nyquist_rad_km"]


def test_wavelength_peak_and_missing_box():
    lat = np.linspace(25, 45, 80)
    lon = np.linspace(150, 170, 80)
    wave = np.broadcast_to(np.sin(2 * np.pi * 8 * np.arange(80) / 80), (80, 80))
    value = module.spectrum(wave, lat, lon, np.ones(wave.shape, bool))
    dx = np.median(np.diff(lon)) * 111.32 * np.cos(np.deg2rad(np.mean(lat)))
    expected = 2 * np.pi * 8 / (80 * dx)
    peak = value["k_rad_km"][int(np.argmax(value["power"]))]
    assert abs(peak - expected) < 0.2 * expected
    assert module.spectrum(wave, lat, lon, np.zeros(wave.shape, bool)) is None


def test_geostrophy_excludes_equator_and_uses_coriolis_sign():
    lat = np.array([-40.0, -20.0, 0.0, 20.0, 40.0])
    lon = np.linspace(140, 160, 10)
    ssh = np.broadcast_to(lat[:, None] * 0.01, (len(lat), len(lon)))
    u, v = module.geostrophic(ssh, lat, lon)
    assert np.isnan(u[2]).all() and np.isnan(v[2]).all()
    assert (u[:2] > 0).all() and (u[3:] < 0).all()
    np.testing.assert_allclose(v[[0, 1, 3, 4]], 0, atol=1e-12)
