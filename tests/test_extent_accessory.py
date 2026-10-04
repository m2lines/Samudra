# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import zarr  # type: ignore[import-untyped]
from torch import nn
from torch.utils.checkpoint import checkpoint

from samudra.experiments.extent_accessory import VarianceTargets, transform_targets
from samudra.experiments.extent_models import AxialBlock, ExtentEvolution
from samudra.experiments.observation_pilot import digest
from samudra.experiments.surface_state import geographic_features


def test_target_controls_separate_seasonal_structure_from_time_varying_anomalies():
    dates = [
        f"{year}-{month:02d}-03" for year in (2000, 2001) for month in range(1, 13)
    ]
    geography = np.array([[0.0, 2.0], [3.0, 0.0]], dtype="f4")
    season = np.tile(np.arange(12, dtype="f4"), 2)[:, None, None]
    perturbation = np.repeat([-0.25, 0.25], 12).astype("f4")[:, None, None]
    values = geography[None] + season + perturbation
    values[:, 1, 1] = 0  # An invalid cell remains zero in every mode.
    seasonal = transform_targets(values, dates, "seasonal")
    anomaly = transform_targets(values, dates, "anomaly")
    np.testing.assert_array_equal(seasonal + anomaly, values)
    np.testing.assert_array_equal(seasonal[:12], seasonal[12:])
    np.testing.assert_array_equal(anomaly[:12], -anomaly[12:])
    assert np.count_nonzero(anomaly[:, 0, 0]) == 24
    static = transform_targets(values, dates, "static")
    np.testing.assert_array_equal(static[0], static[-1])
    np.testing.assert_array_equal(static[0], [[5.5, 7.5], [8.5, 0]])
    for mode in ("aligned", "static", "seasonal", "shuffled", "anomaly"):
        assert not transform_targets(values, dates, mode)[:, 1, 1].any()


def test_shuffled_target_preserves_maps_and_global_rng_but_breaks_date_alignment():
    values = np.arange(96, dtype="f4").reshape(24, 2, 2)
    dates = [f"2000-{1 + i % 12:02d}-03" for i in range(24)]
    np.random.seed(1729)
    expected_random = np.random.random(4)
    np.random.seed(1729)
    first = transform_targets(values, dates, "shuffled")
    np.testing.assert_array_equal(np.random.random(4), expected_random)
    np.testing.assert_array_equal(first, transform_targets(values, dates, "shuffled"))
    assert not np.array_equal(first, values)
    np.testing.assert_array_equal(first[np.argsort(first[:, 0, 0])], values)


def test_auxiliary_head_preserves_core_initialization_and_routes_gradients(monkeypatch):
    import samudra.experiments.extent_models as models

    monkeypatch.setattr(
        models,
        "make_unet",
        lambda i, o, widths: nn.Sequential(
            nn.Conv2d(i, widths[0], 1), nn.Conv2d(widths[0], o, 1)
        ),
    )
    torch.manual_seed(1729)
    control = ExtentEvolution(2)
    next_random = torch.rand(4)
    torch.manual_seed(1729)
    auxiliary = ExtentEvolution(2, variant="aux")
    torch.testing.assert_close(torch.rand(4), next_random, rtol=0, atol=0)
    for name, value in control.state_dict().items():
        torch.testing.assert_close(auxiliary.state_dict()[name], value, rtol=0, atol=0)
    states = torch.randn(1, 2, 2, 8, 12, requires_grad=True)
    forcing = torch.randn(1, 1, 3, 8, 12)
    geo = geographic_features(torch.linspace(-20, 20, 8), torch.arange(12).float())
    context = torch.cat((geo, torch.zeros(2, 8, 12)))[None]
    mask = torch.ones(2, 8, 12, dtype=torch.bool)
    args = (states, forcing, context, mask, 1, "om4")
    physical, small = checkpoint(auxiliary, *args, True, use_reentrant=False)
    torch.testing.assert_close(physical, control(*args), rtol=0, atol=0)
    small.square().mean().backward()
    assert isinstance(auxiliary.net, nn.Sequential)
    assert auxiliary.auxiliary_head.weight.grad is not None
    assert auxiliary.net[0].weight.grad is not None
    assert auxiliary.net[1].weight.grad is None
    auxiliary.zero_grad(set_to_none=True)
    auxiliary(*args).square().mean().backward()
    assert auxiliary.auxiliary_head.weight.grad is None


@pytest.mark.parametrize("shape", [(5, 9), (8, 8)])
def test_axial_block_accepts_dynamic_extents_and_mixes_both_axes(shape):
    torch.manual_seed(4)
    block = AxialBlock(16, heads=4)
    inputs = torch.randn(1, 16, *shape, requires_grad=True)
    out = checkpoint(block, inputs, use_reentrant=False)
    assert out.shape == inputs.shape and torch.isfinite(out).all()
    out[0, 0, 0, 0].backward()
    assert inputs.grad is not None and inputs.grad[0, :, -1, -1].abs().sum() > 0
    for attention in block.attention:
        assert attention.in_proj_weight.grad is not None


def test_accessory_uses_future_labels_exact_dates_and_training_transform(tmp_path):
    import datetime

    dates = np.array(
        [
            str(datetime.date(2000, 1, 1) + datetime.timedelta(days=5 * i))
            for i in range(30)
        ]
    )
    shape = (3, 4)
    valid = np.ones(shape, bool)
    valid[0, 0] = False
    np.savez(
        tmp_path / "grid.npz",
        dates=dates,
        lat=np.arange(3),
        lon=np.arange(4),
        mask=valid,
        area=np.ones(shape),
    )
    (tmp_path / "manifest.json").write_text("{}")
    norm = dict(variance_scale=2.0, log_mean=0.5, log_std=0.2)
    (tmp_path / "normalization.json").write_text(json.dumps(norm))
    raw = np.broadcast_to(np.arange(30, dtype="f4")[:, None, None], (30, *shape)).copy()
    raw[:, 0, 0] = np.nan
    store = zarr.open_group(str(tmp_path / "targets.zarr"), mode="w")
    store.create_dataset("variance", data=raw)
    store.create_dataset("verified", data=np.ones(30, bool))
    zarr.consolidate_metadata(store.store)
    ready = {
        name + "_sha256": digest(
            tmp_path / (name + (".npz" if name == "grid" else ".json"))
        )
        for name in ("grid", "manifest", "normalization")
    }
    (tmp_path / "READY.json").write_text(json.dumps(dict(ready, frames=30)))
    data = SimpleNamespace(
        device="cpu",
        trainset=SimpleNamespace(
            sources=[SimpleNamespace(time=SimpleNamespace(values=dates))]
        ),
        mask=torch.ones(2, *shape),
        lat=torch.arange(3),
        lon=torch.arange(4),
    )
    targets = VarianceTargets(tmp_path, data, 0.01)
    for lead in (1, 6):
        expected = (np.log1p((2 + 18 + lead) / 2) - 0.5) / 0.2
        prediction = torch.full((1, 1, *shape), expected, requires_grad=True)
        assert targets.loss(prediction, [2], lead).item() < 1e-10
        targets.loss(prediction + 1, [2], lead).backward()
        assert prediction.grad is not None and prediction.grad[0, 0, 0, 0] == 0
        assert prediction.grad[0, 0, 1, 1] > 0
    (tmp_path / "normalization.json").write_text("{}")
    with pytest.raises(ValueError, match="hash differs"):
        VarianceTargets(tmp_path, data, 0.01)
