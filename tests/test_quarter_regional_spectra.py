# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Check support and averaging semantics of native-grid spectral diagnostics."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "quarter_regional_spectra", SCRIPTS / "quarter_regional_spectra.py"
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_fine_hole_excludes_whole_common_cell_without_interpolating_values():
    coarse = (np.array([0.5, 1.5]), np.array([0.5, 1.5]))
    fine = (np.array([0.25, 0.75, 1.25, 1.75]), np.array([0.25, 0.75, 1.25, 1.75]))
    cmask = np.ones((2, 2), dtype=bool)
    fmask = np.ones((4, 4), dtype=bool)
    fmask[0, 0] = False
    shared, projected = module.common_support([cmask, fmask], [coarse, fine], 0)
    np.testing.assert_array_equal(shared, [[False, True], [True, True]])
    expected = np.ones((4, 4), dtype=bool)
    expected[:2, :2] = False
    np.testing.assert_array_equal(projected[1], expected)
    np.testing.assert_array_equal(projected[0], shared)


def test_phase_cancellation_distinguishes_mean_power_from_power_of_mean():
    lat, lon = np.linspace(25, 45, 80), np.linspace(150, 170, 80)
    wave = np.broadcast_to(np.sin(2 * np.pi * 8 * np.arange(80) / 80), (80, 80))
    fields = np.stack([wave, -wave, np.zeros_like(wave)])[:, None]
    result = module.aggregate_spectra(fields, lat, lon, np.ones((80, 80), bool))
    single = module.spectrum(wave, lat, lon, np.ones((80, 80), bool))
    np.testing.assert_allclose(
        result["snapshot_power"], np.asarray(single["power"]) * 2 / 3
    )
    np.testing.assert_allclose(result["mean_field"], 0, atol=1e-20)
    assert len(result["per_origin_power"]) == 3


def test_velocity_spectrum_is_half_sum_of_component_powers():
    lat, lon = np.linspace(25, 45, 40), np.linspace(150, 170, 40)
    rng = np.random.default_rng(12)
    fields = rng.normal(size=(3, 2, 40, 40))
    support = np.ones((40, 40), bool)
    result = module.aggregate_spectra(fields, lat, lon, support)
    expected = np.mean(
        [
            [module.spectrum(f, lat, lon, support)["power"] for f in values]
            for values in fields
        ],
        axis=(0, 1),
    )
    np.testing.assert_allclose(result["snapshot_power"], expected)


def test_native_coarse_nyquist_is_not_extended_to_fine_grid():
    curves = []
    for cells in (20, 80):
        lat, lon = np.linspace(25, 45, cells), np.linspace(150, 170, cells)
        fields = np.random.default_rng(2).normal(size=(3, 1, cells, cells))
        curves.append(
            module.aggregate_spectra(fields, lat, lon, np.ones((cells, cells), bool))
        )
    assert max(curves[0]["k_rad_km"]) < curves[0]["nyquist_rad_km"]
    assert max(curves[0]["k_rad_km"]) < max(curves[1]["k_rad_km"])
    np.testing.assert_allclose(
        curves[1]["nyquist_rad_km"] / curves[0]["nyquist_rad_km"], 79 / 19
    )


def test_target_dates_cannot_be_substituted_by_same_nominal_lead():
    import pytest

    start = np.array(
        [
            ["2015-01-26", "2015-12-27"],
            ["2018-01-26", "2018-12-27"],
            ["2021-01-26", "2021-12-27"],
        ],
        dtype="datetime64[h]",
    )
    good = dict(
        bin_start=start,
        bin_end=start + np.timedelta64(5, "D"),
        bin_midpoint=start + np.timedelta64(60, "h"),
    )
    bad = dict(good, bin_midpoint=start + np.timedelta64(61, "h"))
    module.validate_dates({"a": good, "b": good})
    with pytest.raises(ValueError, match="dates differ"):
        module.validate_dates({"a": good, "b": bad})
    shifted = {k: v + np.timedelta64(1, "D") for k, v in good.items()}
    with pytest.raises(ValueError, match="requested day30/day365"):
        module.validate_dates({"a": shifted, "b": shifted})


def test_support_is_identical_at_both_leads(monkeypatch):
    monkeypatch.setattr(module, "REGIONS", [("Kuroshio", (150, 170), (25, 45))])
    lat, lon = np.linspace(25, 45, 20), np.linspace(150, 170, 20)
    values = np.random.default_rng(4).normal(size=(3, 2, 2, 20, 20))
    values[2, 1, 0, 10, 10] = np.nan
    source = dict(surface=values, lat=lat, lon=lon, mask=np.ones((2, 20, 20), bool))
    result = module.calculate({"source": source})
    a = result["leads"]["30"]["SST"]["Kuroshio"]["curves"]["source"]
    b = result["leads"]["365"]["SST"]["Kuroshio"]["curves"]["source"]
    assert a["shared_valid_cells"] == b["shared_valid_cells"] == 399


def test_plot_ke_with_empty_equatorial_region(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "VARIABLES", {"Geostrophic KE": (1, "m² s⁻² km")})
    lat, lon = np.arange(-59.0, 60, 2), np.arange(1.0, 360, 2)
    source = dict(
        surface=np.random.default_rng(4).normal(size=(3, 2, 2, len(lat), len(lon))),
        lat=lat,
        lon=lon,
        mask=np.ones((2, len(lat), len(lon)), bool),
    )
    result = module.calculate({f"source-{i}": source for i in range(5)})
    module.plot(result, tmp_path)
    assert len(list(tmp_path.glob("*geostrophic-ke.png"))) == 4
