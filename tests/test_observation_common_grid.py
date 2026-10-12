# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Common reporting geometry must never acquire forecast-dependent support."""

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from samudra.experiments.observation_common_grid import CommonGrid, reporting_masks
from samudra.experiments.observation_metrics import score, selection_score


def grid(lat=None, lon=None):
    lat = np.linspace(-85, 85, 18) if lat is None else np.asarray(lat)
    lon = np.arange(16) * 22.5 if lon is None else np.asarray(lon)
    return dict(lat=lat, lon=lon, mask=np.ones((77, len(lat), len(lon)), dtype=bool))


def edges(a, latitude=False):
    b = np.r_[
        a[0] - (a[1] - a[0]) / 2, (a[:-1] + a[1:]) / 2, a[-1] + (a[-1] - a[-2]) / 2
    ]
    if latitude:
        b[0], b[-1] = -90, 90
        b = np.sin(np.deg2rad(b))
    return b


def independent_mean(source, target, field, wet):
    sy, sx = edges(source["lat"], True), edges(source["lon"])
    ty, tx = edges(target["lat"], True), edges(target["lon"])
    result, coverage = (
        np.zeros((len(ty) - 1, len(tx) - 1)),
        np.zeros((len(ty) - 1, len(tx) - 1)),
    )
    for y, x in np.ndindex(result.shape):
        numerator = denominator = 0.0
        for j, k in np.ndindex(field.shape):
            dy = max(0, min(ty[y + 1], sy[j + 1]) - max(ty[y], sy[j]))
            dx = sum(
                max(0, min(tx[x + 1], sx[k + 1] + shift) - max(tx[x], sx[k] + shift))
                for shift in (-360, 0, 360)
            )
            weight = dy * dx * wet[j, k]
            numerator += weight * (field[j, k] if wet[j, k] else 0)
            denominator += weight
        result[y, x] = numerator / denominator if denominator else np.nan
        coverage[y, x] = denominator / ((ty[y + 1] - ty[y]) * (tx[x + 1] - tx[x]))
    return result, coverage


def test_non_nested_periodic_wet_means_match_independent_overlap():
    source = grid([-81, -52, -19, 14, 49, 80], [5, 65, 125, 185, 245, 305])
    target = grid([-70, -21, 31, 78], [40, 130, 220, 310])
    source["mask"][:, 1, 0] = False
    geometry = CommonGrid(source, target, min_coverage=0.4)
    values = np.arange(36).reshape(6, 6).astype(float)
    expected, coverage = independent_mean(source, target, values, source["mask"][38])
    actual = geometry.project(np.stack([values, values * 2]))
    expected[coverage < 0.4 - 1e-10] = np.nan
    np.testing.assert_allclose(actual[0], expected, rtol=1e-7)
    np.testing.assert_allclose(actual[1], expected * 2, rtol=1e-7)
    np.testing.assert_allclose(geometry.coverage[0], coverage)
    assert coverage[:, -1].min() < 1  # Wrapped source cell crosses target seam.


def test_full_globe_area_conservation_and_constant_field():
    source = grid(np.linspace(-87, 87, 30), np.arange(48) * 7.5 + 3.75)
    target = grid()
    geometry = CommonGrid(source, target)
    rng = np.random.default_rng(8)
    values = rng.normal(size=(2, 30, 48))
    result = geometry.project(values)
    source_area = np.diff(edges(source["lat"], True))[:, None] * np.diff(
        edges(source["lon"])
    )
    np.testing.assert_allclose(
        (result * geometry.remap.area).sum((-2, -1)),
        (values * source_area).sum((-2, -1)),
        atol=2e-6,
    )
    np.testing.assert_allclose(geometry.project(np.ones_like(values)), 1)


def test_fixed_channel_support_and_nonfinite_rejection():
    source, target = grid(), grid()
    source["mask"][76, 5, 6] = False
    source["mask"][49, 6, 7] = False  # 1050 m affects only the deep OHC layer.
    geometry = CommonGrid(source, target)
    assert not geometry.surface_support[5, 6]
    assert geometry.ohc_support[0, 6, 7]
    assert not geometry.ohc_support[1, 6, 7]
    values = np.ones((3, 2, 18, 16))
    values[:, 1, 5, 6] = np.nan
    before = geometry.coverage.copy()
    result = geometry.project(values)
    assert np.isnan(result[:, 1, 5, 6]).all()
    values[0, 0, 8, 8] = np.inf
    with pytest.raises(ValueError, match="fixed source wet"):
        geometry.project(values)
    np.testing.assert_array_equal(before, geometry.coverage)
    with pytest.raises(ValueError, match="source geometry"):
        geometry.validate_source(target)


def monthly_arrays():
    rng = np.random.default_rng(82)
    return dict(
        prediction=rng.normal(size=(3, 6, 2, 18, 16)),
        predicted_ohc=rng.normal(size=(3, 2, 18, 16)),
        reference=rng.normal(size=(3, 6, 4, 18, 16)),
        reference_ohc=rng.normal(size=(3, 2, 18, 16)),
        climatology_surface=rng.normal(size=(3, 6, 2, 18, 16)),
        climatology_ohc=rng.normal(size=(3, 2, 18, 16)),
    )


def test_identity_scoring_preserves_legacy_and_rescores_same_climatology():
    geometry = CommonGrid(grid(), grid())
    arrays = monthly_arrays()
    result = geometry.score(**arrays)
    kwargs = dict(
        reference=arrays["reference"],
        reference_ohc=arrays["reference_ohc"],
        lat=geometry.lat,
        lon=geometry.lon,
        mask=geometry.surface_support,
        global_observations=True,
        strict_velocity_support=True,
        compute_spectra=False,
        climatology_surface=arrays["climatology_surface"],
    )
    legacy = score(
        prediction=arrays["prediction"], predicted_ohc=arrays["predicted_ohc"], **kwargs
    )
    control = score(
        prediction=arrays["climatology_surface"],
        predicted_ohc=arrays["climatology_ohc"],
        **kwargs,
    )
    assert result["candidate"] == legacy
    assert result["control"] == control
    assert result["common_grid_rmse_ratio"] == selection_score(
        legacy, control, [], metric="rmse-only"
    )
    np.testing.assert_array_equal(
        geometry.project(arrays["prediction"]), arrays["prediction"].astype("f4")
    )


def test_restricted_intersection_changes_control_not_just_candidate():
    arrays = monthly_arrays()
    unrestricted = CommonGrid(grid(), grid()).score(**arrays)
    source = grid()
    source["mask"][:, :, :4] = False
    geometry = CommonGrid(source, grid())
    restricted = geometry.score(**arrays)
    assert (
        restricted["control"]["metrics"]["sst_rmse"]
        != unrestricted["control"]["metrics"]["sst_rmse"]
    )
    bad = arrays["prediction"].copy()
    bad[0, 0, 0, 8, 8] = np.nan
    with pytest.raises(ValueError, match="Nonfinite prediction"):
        geometry.score(**{**arrays, "prediction": bad})
    # Canonical missing truth is shared, not manufactured from a model output.
    arrays["reference"][:, :, :, 8, 8] = np.nan
    with pytest.raises(ValueError, match="Nonfinite prediction"):
        geometry.score(**{**arrays, "prediction": bad})


@pytest.mark.parametrize(
    "steps,months", [(6, ()), (73, tuple(f"2015-{m:02d}" for m in range(1, 13)))]
)
def test_sidecar_uses_pre_iap_maps_and_preserves_input(tmp_path, steps, months):
    geometry = CommonGrid(grid(), grid())
    surface = np.ones((steps, 2, 18, 16), dtype="f4")
    ohc = np.full(((12,) if months else ()) + (2, 18, 16), 7, dtype="f4")
    path = tmp_path / "sidecar" / "2015-01.npz"
    geometry.write_sidecar(
        path, surface=surface, model_ohc=ohc, origins=["2015-01"], months=months
    )
    with np.load(path) as saved:
        np.testing.assert_array_equal(saved["prediction"], surface)
        np.testing.assert_array_equal(saved["predicted_ohc"], ohc)
        np.testing.assert_array_equal(saved["coverage"], geometry.coverage)
        assert saved["origins"].tolist() == ["2015-01"]
        assert saved["months"].tolist() == list(months)
    assert np.all(ohc == 7)


def test_annual_scoring_keeps_temporal_eke_and_monthly_ohc(monkeypatch):
    from samudra.experiments import observation_annual as annual

    monkeypatch.setattr(annual, "spatial_error", lambda *args: None)
    geometry = CommonGrid(grid(), grid())
    rng = np.random.default_rng(10)
    prediction = rng.normal(size=(73, 2, 18, 16))
    reference = rng.normal(size=(73, 4, 18, 16))
    months = [f"2015-{m:02d}" for m in range(1, 13)]
    result = geometry.score_annual(
        prediction,
        np.ones((12, 2, 18, 16)),
        reference=reference,
        reference_ohc=np.zeros((12, 2, 18, 16)),
        months=months,
    )
    expected = annual.surface_metrics(
        prediction, reference, SimpleNamespace(grid=grid(), global_observations=True)
    )
    assert result["annual_eke_rmse"] == expected["annual_eke_rmse"]
    assert result["leads"] == expected["leads"]
    assert result["monthly_ohc"]["2015-12"] == {"0_700": 1, "700_2000": 1}


def test_masks_require_physical_canonical_grid():
    bad = grid()
    bad["mask"] = bad["mask"][:28]
    with pytest.raises(ValueError, match="77-channel"):
        reporting_masks(bad)


def test_pilot_sidecar_uses_full_calendar_weights_before_iap_projection(
    tmp_path, monkeypatch
):
    import torch

    from samudra.experiments import observation_pilot as pilot_module

    model_grid = grid()
    model_grid["names"] = np.asarray([str(i) for i in range(77)])
    geometry = CommonGrid(model_grid, grid())
    prediction = torch.arange(1, 8, dtype=torch.float32)[
        None, :, None, None, None
    ].expand(1, 7, 77, 18, 16)
    weights = torch.tensor([5, 5, 5, 5, 5, 5, 1], dtype=torch.float32) / 31
    sample = dict(
        name="2015-01",
        month_weights=weights,
        interior=torch.zeros(1, 28, 18, 16),
        raw=dict(
            surface=np.zeros((26, 2, 18, 16)),
            velocity=np.zeros((26, 2, 18, 16)),
            ohc=np.zeros((2, 18, 16)),
        ),
    )
    projected_iap = np.full((1, 2, 18, 16), 999, dtype="f4")
    data = SimpleNamespace(
        grid=model_grid,
        load=lambda path: sample,
        dual_grid=False,
        evaluation_fields=lambda p, s: dict(
            surface=p[:, :, :2], interior=torch.zeros(1, 28, 18, 16), ohc=projected_iap
        ),
        model_ohc=lambda state: (state[:, :2] * 10).numpy(),
        climatology_fields=lambda s: dict(surface=torch.zeros(1, 7, 2, 18, 16)),
        interior_support=torch.ones(28, 18, 16, dtype=torch.bool),
        interior_area=torch.ones(18, 16),
        ts_indices=list(range(28)),
        ohc_metric_kwargs={},
    )
    pilot: Any = pilot_module.Pilot.__new__(pilot_module.Pilot)
    pilot.data = data
    pilot.args = SimpleNamespace(selection_metric="rmse-only")
    pilot.model = SimpleNamespace(
        eval=lambda: None, forecast=lambda: (prediction, None)
    )
    pilot.arguments = lambda s: ()
    monkeypatch.setattr(
        pilot_module, "score", lambda **kwargs: dict(metrics={}, spectra={})
    )
    monkeypatch.setattr(pilot_module.torch, "autocast", lambda *a, **k: nullcontext())
    baseline = pilot.evaluate([tmp_path / "2015-01.npz"])
    actual = pilot.evaluate(
        [tmp_path / "2015-01.npz"],
        common_grid=geometry,
        common_export=tmp_path / "common",
    )
    assert actual == baseline
    with np.load(tmp_path / "common" / "2015-01.npz") as saved:
        expected = float((torch.arange(1, 8) * weights).sum() * 10)
        np.testing.assert_allclose(saved["predicted_ohc"], expected)
        assert not np.any(saved["predicted_ohc"] == 999)
        assert saved["prediction"].shape == (6, 2, 18, 16)
    with pytest.raises(ValueError, match="frozen legacy climatology"):
        pilot.evaluate(
            [], climatology=True, common_grid=geometry, common_export=tmp_path
        )


def test_velocity_is_derived_after_coarsening_not_averaged_before():
    from samudra.experiments.observation_metrics import _field, weighted_rmse
    from samudra.metrics.kernels import geostrophic_velocity_from_zos

    source = grid(np.linspace(-87, 87, 36), np.arange(32) * 11.25)
    geometry = CommonGrid(source, grid())
    rng = np.random.default_rng(5)
    physical = rng.normal(size=(3, 6, 2, 36, 32))
    arrays = monthly_arrays()
    arrays["prediction"] = geometry.project(physical)
    result = geometry.score(**arrays)
    pu, pv = geostrophic_velocity_from_zos(
        _field(arrays["prediction"][:, 5, 1], geometry.lat, geometry.lon), "lat", "lon"
    )
    velocity = np.stack([a.transpose("time", "lat", "lon").values for a in (pu, pv)], 1)
    observed = arrays["reference"][:, 5, 2:].copy()
    observed[:, :, np.abs(geometry.lat) < 5] = np.nan
    expected = np.sqrt(2) * weighted_rmse(velocity, observed, geometry.area)
    assert result["candidate"]["metrics"]["velocity_rmse/day30"] == pytest.approx(
        expected
    )
    fine_u, _ = geostrophic_velocity_from_zos(
        _field(physical[:, 5, 1], source["lat"], source["lon"]),
        "lat",
        "lon",
        equatorial_mask_deg=0,
    )
    fine_values = fine_u.transpose("time", "lat", "lon").values
    wrong = geometry.project(np.stack([fine_values, fine_values], 1))[:, 0]
    assert not np.allclose(wrong[:, :5], velocity[:, 0, :5])


@pytest.mark.parametrize("persistence", [False, True])
def test_annual_sidecar_inventory_and_hashes(tmp_path, persistence):
    import hashlib

    from samudra.experiments.observation_common_grid import sidecar_hashes

    origins = ["2015-01-01", "2018-01-01", "2021-01-01"]
    expected = [f"{origin}.npz" for origin in origins]
    if persistence:
        expected += [f"{origin}-persistence.npz" for origin in origins]
    for name in expected:
        (tmp_path / name).write_bytes(name.encode())
    receipt = sidecar_hashes(tmp_path, expected)
    assert receipt == {
        name: hashlib.sha256(name.encode()).hexdigest() for name in expected
    }
    (tmp_path / expected[-1]).unlink()
    with pytest.raises(ValueError, match="missing="):
        sidecar_hashes(tmp_path, expected)
    (tmp_path / expected[-1]).write_bytes(expected[-1].encode())
    (tmp_path / "unexpected.npz").touch()
    with pytest.raises(ValueError, match="unexpected=.*unexpected.npz"):
        sidecar_hashes(tmp_path, expected)
