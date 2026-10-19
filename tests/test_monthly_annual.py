# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from samudra.experiments.monthly_annual import lead_metrics, month_layout
from samudra.experiments.observation_metrics import _field
from samudra.metrics import kernels


@pytest.mark.parametrize("npz_strings", [False, True])
def test_month_layout_covers_calendar_and_keeps_absolute_lead(npz_strings):
    dates = pd.date_range("2015-01-01", periods=73, freq="5D")
    starts = (
        dates.strftime("%Y-%m-%dT%H:%M:%S").to_numpy(dtype="U19")
        if npz_strings
        else dates
    )
    jan, w, d, p = month_layout(starts, "2015-01")
    np.testing.assert_array_equal(jan, np.arange(7))
    np.testing.assert_allclose(w, np.array([5] * 6 + [1]) / 31)
    assert d == 31 and p == 15.5
    dec, w, d, p = month_layout(starts, "2015-12")
    assert dec[-1] == 72 and p == 349.5
    assert w.sum() == pytest.approx(1)
    with pytest.raises(ValueError, match="Incomplete calendar month"):
        month_layout(starts[:6], "2015-01")


def test_fixed_lead_metrics_zero_for_perfect_case_predictions():
    lat = np.linspace(10, 70, 12)
    lon = np.arange(0, 360, 20)
    lonplane, latplane = np.meshgrid(np.deg2rad(lon), np.deg2rad(lat))
    frames = np.stack(
        [np.sin(lonplane) * np.cos(latplane) * scale for scale in [0.1, 0.2, 0.4]]
    )
    pu, pv = kernels.geostrophic_velocity_from_zos(
        _field(frames, lat, lon), "lat", "lon"
    )
    u, v = [a.transpose("time", "lat", "lon").values for a in [pu, pv]]
    surface = np.stack([frames + 15, frames], axis=1)
    prediction = np.repeat(surface[:, None], 73, axis=1)
    reference = np.repeat(
        np.concatenate([surface, u[:, None], v[:, None]], 1)[:, None], 73, axis=1
    )
    data = SimpleNamespace(
        grid={"lat": lat, "lon": lon, "mask": np.ones((77, len(lat), len(lon)), bool)}
    )
    result = lead_metrics(prediction, reference, data)
    assert len(result) == 73 and result[-1]["day"] == 365
    assert all(
        r[k] == pytest.approx(0)
        for r in result
        for k in ["sst_rmse", "adt_rmse", "velocity_rmse", "eke_rmse"]
    )


@pytest.mark.parametrize("mask_dtype", [torch.bool, torch.float32])
def test_annual_history_accepts_samples_float_validity(mask_dtype):
    from samudra.experiments.monthly_annual import history_fields

    surface = torch.randn(1, 19, 2, 3, 4)
    valid = torch.ones_like(surface, dtype=mask_dtype)
    valid[:, :, 0, 0, 0] = 0
    fields, available = history_fields(surface, valid)
    assert fields.dtype == surface.dtype and available.dtype == torch.bool
    torch.testing.assert_close(fields[:, :, [38, 76]], surface, rtol=0, atol=0)
    torch.testing.assert_close(available[:, :, [38, 76]], valid.bool(), rtol=0, atol=0)
    assert not available[:, :, :38].any()
    with pytest.raises(ValueError, match="19-bin"):
        history_fields(surface[:, :18], valid[:, :18])


def test_full_annual_readout_with_saved_string_timestamps(tmp_path):
    from samudra.experiments.monthly_annual import rollout
    from samudra.experiments.monthly_latent import MonthlyLatentOcean

    h, w = 3, 4
    lat, lon = np.linspace(-60, 60, h), np.arange(w) * 90.0
    midpoints = (
        pd.date_range("2015-01-01", periods=92, freq="5D")
        - pd.Timedelta(days=95)
        + pd.Timedelta(days=2.5)
    )
    starts = (
        (midpoints - pd.Timedelta(days=2.5))
        .strftime("%Y-%m-%dT%H:%M:%S")
        .to_numpy(dtype="U19")
    )
    raw = dict(
        surface=np.ones((92, 2, h, w), np.float32),
        atmosphere=np.ones((92, 8, h, w), np.float32),
        start=starts,
        midpoint=midpoints.strftime("%Y-%m-%dT%H:%M:%S").to_numpy(dtype="U19"),
    )
    grid = dict(
        lat=lat,
        lon=lon,
        mask=np.ones((77, h, w), bool),
        mean=np.zeros(77, np.float32),
        std=np.ones(77, np.float32),
    )
    stats = dict(
        surface_climatology=np.zeros((12, 2, h, w), np.float32),
        atmosphere_mean=np.zeros(8, np.float32),
        atmosphere_std=np.ones(8, np.float32),
    )
    data = SimpleNamespace(
        root=tmp_path,
        device="cpu",
        grid=grid,
        stats=stats,
        surface_fill="zero",
        tensor=lambda a: torch.as_tensor(a, dtype=torch.float32),
        mask=torch.ones(77, h, w),
        ts_indices=list(range(38, 52)) + list(range(57, 71)),
        ts_mask=torch.ones(28, h, w),
        mean=torch.zeros(1, 1, 77, 1, 1),
        std=torch.ones(1, 1, 77, 1, 1),
        load=lambda _: {"atmosphere": torch.zeros(1, 26, 8, h, w)},
        ohc=lambda x: x[:, [38, 39]].numpy(),
    )
    calls = []

    def sample(latent, support, generator, *, steps, position_days, duration_days=5.0):
        calls.append((position_days, duration_days))
        return (latent[:, 0, 0, 0, 0, None, None, None] + position_days / 365.0).expand(
            -1, 77, h, w
        ) * support

    def trajectory(fields, available, forcing, contexts):
        assert available.dtype == torch.bool
        assert fields.shape[1] == 19 and forcing.shape[1] == 92
        initial = fields.mean().reshape(1, 1, 1, 1, 1)
        return [initial + i / 100 for i in range(74)]

    model = SimpleNamespace(
        shape=(h, w),
        adapt=lambda x: x[:, :, :3],
        trajectory=trajectory,
        decoder=SimpleNamespace(sample=sample),
        monthly_condition=MonthlyLatentOcean.monthly_condition,
    )
    description = dict(
        origin="2015-01-01",
        monthly_interiors=[
            dict(file=f"2015-{i:02d}-interior.npz") for i in range(1, 13)
        ],
    )
    result = rollout(model, data, description, raw, origin_index=0)
    assert result["surface"].shape == (73, 2, h, w)
    assert result["ohc"].shape == (12, 2, h, w)
    assert result["latent_rms"].shape == (74,)
    assert len(calls) == 73 + 12
    assert calls[-1] == (349.5, 31.0)
    assert np.isfinite(result["surface"]).all() and np.isfinite(result["ohc"]).all()
    raw["surface"][19:] = 999
    result2 = rollout(model, data, description, raw, origin_index=0)
    np.testing.assert_array_equal(result2["surface"], result["surface"])
    np.testing.assert_array_equal(result2["ohc"], result["ohc"])
