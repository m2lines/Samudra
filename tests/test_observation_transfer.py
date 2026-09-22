# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from samudra.experiments import initializer_models
from samudra.experiments.observation_metrics import PROTOCOL, selection_score


def test_per_frame_validity_retains_old_path_and_surface_copy(monkeypatch):
    monkeypatch.setattr(
        initializer_models, "make_unet", lambda i, o, w: nn.Conv2d(i, o, 1)
    )
    model = initializer_models.HistoryInitializer(
        ["thetao_0", "so_0", "zos"], "wide", True
    )
    surface = torch.randn(1, 19, 2, 8, 8)
    forcing = torch.randn(1, 19, 3, 8, 8)
    context = torch.randn(1, 5, 8, 8)
    mask = torch.ones(3, 8, 8)
    old = model(surface, forcing, context, mask)
    valid = torch.ones_like(surface)
    new = model(surface, forcing, context, mask, valid)
    torch.testing.assert_close(old, new, rtol=0, atol=0)
    valid[:, 3, :, 4, 4] = 0
    missing = model(surface, forcing, context, mask, valid)
    torch.testing.assert_close(missing[:, :, [0, 2]], surface[:, -2:])
    assert not torch.equal(missing[:, :, 1], old[:, :, 1])
    with pytest.raises(ValueError):
        model(surface, forcing, context, mask, valid[:, :, :1])


def test_spectral_collapse_can_lose_despite_improved_pointwise_errors():
    control = {"metrics": {key: 1.0 for key in PROTOCOL["integrated"]}}
    good: dict[str, Any] = {
        "metrics": {key: 1.0 for key in PROTOCOL["integrated"]},
        "spectra": {"sst/open/day30": {"error_dex": 0.05}},
    }
    collapsed: dict[str, Any] = {
        "metrics": {key: 0.8 for key in PROTOCOL["integrated"]},
        "spectra": {"sst/open/day30": {"error_dex": 1.0}},
    }
    keys = list(good["spectra"])
    assert selection_score(good, control, keys) < selection_score(
        collapsed, control, keys
    )
    with pytest.raises(ValueError):
        selection_score(good, control, [])
    collapsed["spectra"][keys[0]]["error_dex"] = np.nan
    with pytest.raises(ValueError):
        selection_score(collapsed, control, keys)


def test_integrated_scorer_identity_and_nonfinite_prediction():
    import xarray as xr

    from samudra.experiments.observation_metrics import score
    from samudra.metrics import kernels

    lat, lon = np.arange(20.5, 50.5), np.arange(170.5, 200.5)
    y, x = np.meshgrid(lat, lon, indexing="ij")
    pattern = np.sin(x / 3) * np.cos(y / 4)
    amplitude = np.arange(1, 10)[:, None, None]
    ssh = pattern[None] * amplitude * 0.01
    sst = 20 + pattern[None] * amplitude * 0.1
    field = xr.DataArray(
        ssh, dims=("time", "lat", "lon"), coords={"lat": lat, "lon": lon}
    )
    u, v = kernels.geostrophic_velocity_from_zos(field, "lat", "lon")
    u, v = u.transpose("time", "lat", "lon"), v.transpose("time", "lat", "lon")
    surfaces = np.stack([sst, ssh], axis=1)
    reference = np.stack([sst, ssh, u.values, v.values], axis=1)
    prediction = np.repeat(surfaces[:, None], 6, axis=1)
    reference = np.repeat(reference[:, None], 6, axis=1)
    ohc = np.ones((9, 2, len(lat), len(lon))) * 1e9
    mask = np.ones((len(lat), len(lon)), dtype=bool)
    result = score(prediction, reference, ohc, ohc, lat, lon, mask)
    assert result["spectra"]
    assert all(abs(value) < 1e-10 for value in result["metrics"].values())
    assert all(abs(curve["error_dex"]) < 1e-10 for curve in result["spectra"].values())
    prediction[0, 0, 0, 3, 3] = np.nan
    with pytest.raises(ValueError, match="Nonfinite prediction"):
        score(prediction, reference, ohc, ohc, lat, lon, mask)


def test_loss_ignores_missing_labels_and_copied_temperature():
    from samudra.experiments.observation_training import Samples

    data = Samples.__new__(Samples)
    data.area = torch.ones(3, 1)
    data.mean = torch.zeros(1, 1, 77, 1, 1)
    data.std = torch.ones_like(data.mean)
    data.ts_indices = list(range(38, 52)) + list(range(57, 71))
    data.ts_mask = torch.ones(28, 3, 4)
    data.ts_mask[0] = 0
    data.ts_scale = torch.ones(1, 28, 1, 1)
    target = torch.ones(1, 28, 3, 4)
    target[:, 2, 1, 1] = float("nan")
    target[:, 0] = 1e6
    prediction = torch.zeros(1, 77, 3, 4, requires_grad=True)
    loss = data.interior_loss(prediction, {"interior": target})
    torch.testing.assert_close(loss, torch.tensor(1.0))
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[:, 38].count_nonzero() == 0
    assert prediction.grad[:, 40, 1, 1].item() == 0
    assert prediction.grad[:, 57].abs().sum() > 0
