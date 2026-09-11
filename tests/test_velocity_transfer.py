# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pytest
import torch
import xarray as xr

from samudra.experiments.velocity_transfer.data import (
    VelocitySource,
    eligible_anchors,
    geometry,
)
from samudra.experiments.velocity_transfer.model import (
    VelocitySamudra,
    rollout,
    weighted_loss,
)
from samudra.experiments.velocity_transfer.prepare import velocity_view


def test_splits_exclude_full_history_forecast_and_gaps():
    times: np.ndarray = np.arange(
        np.datetime64("2018-01-01"), np.datetime64("2023-01-01"), np.timedelta64(5, "D")
    )
    from samudra.experiments.velocity_transfer.prepare import SPLITS

    for split, (start, end) in SPLITS.items():
        anchors = eligible_anchors(times, split)
        assert len(anchors) > 0
        assert (times[anchors - 3] >= np.datetime64(start)).all()
        assert (times[anchors + 6] <= np.datetime64(end)).all()
    gap = np.delete(times, 10)
    assert 9 not in eligible_anchors(gap, "train")
    assert 10 not in eligible_anchors(gap, "train")


def test_geostrophic_auxiliary_uses_gradients_and_masks_coasts():
    lat = np.array([-30.0, -15.0, 0.0, 15.0, 30.0])
    lon = np.arange(8) * 45.0
    ssh = np.broadcast_to(lat[:, None] / 100, (5, 8)).copy()[None]
    mask = np.ones((5, 8), dtype=bool)
    mask[3, 3] = False
    ds = xr.Dataset(
        {"zos": (("time", "y", "x"), ssh), "mask_0": (("y", "x"), mask)},
        coords={"time": [np.datetime64("2015-01-01")], "y": lat, "x": lon},
    )
    actual = velocity_view(ds, "om4")
    assert actual.u.sel(y=-30).mean() > 0
    assert actual.u.sel(y=30).mean() < 0
    np.testing.assert_allclose(actual.v.values[np.isfinite(actual.v)], 0)
    assert np.isnan(actual.u.sel(y=0)).all()
    assert np.isnan(actual.u.isel(y=4, x=3)).all()


def test_duacs_weighted_coarsening_preserves_constant_and_missing_data():
    data = np.ones((1, 4, 8), dtype=np.float32)
    data[0, 0, 0] = np.nan
    ds = xr.Dataset(
        {
            "ugos": (("time", "latitude", "longitude"), data * 2),
            "vgos": (("time", "latitude", "longitude"), data * -3),
        },
        coords={
            "time": [np.datetime64("2015-01-01")],
            "latitude": [-50.0, -40.0, 40.0, 50.0],
            "longitude": np.arange(8) * 45 - 180,
        },
    )
    result = velocity_view(ds, "duacs")
    assert result.u.shape == (1, 2, 4)
    np.testing.assert_allclose(result.u.values[np.isfinite(result.u)], 2)
    np.testing.assert_allclose(result.v.values[np.isfinite(result.v)], -3)
    assert np.isnan(result.u.values).sum() == 1
    assert (np.diff(result.x) > 0).all()


def synthetic_batch(height=32, width=48, steps=2):
    offsets = torch.arange(-3, steps + 1)[None].float() * 5
    return {
        "history": torch.randn(1, 4, 2, height, width),
        "targets": torch.randn(1, steps, 2, height, width),
        "history_valid": torch.ones(1, 4, 1, height, width, dtype=torch.bool),
        "target_valid": torch.ones(1, steps, 1, height, width, dtype=torch.bool),
        "static": torch.randn(1, 8, height, width),
        "area": torch.ones(1, 1, height, width),
        "day_offsets": offsets,
        "year_day": torch.zeros(1),
    }


def test_shared_backbone_gradients_and_checkpointed_topology_match():
    torch.set_num_threads(2)
    torch.manual_seed(4)
    model = VelocitySamudra((8, 12), checkpointing="all").eval()
    # Nonzero heads make both source losses exercise the entire shared backbone.
    for head in model.heads.values():
        assert isinstance(head, torch.nn.Conv2d)
        torch.nn.init.normal_(head.weight, std=0.1)
    reference = VelocitySamudra((8, 12), checkpointing=None).eval()
    reference.load_state_dict(model.state_dict())
    for source, periodic in (("duacs", True), ("om4", False)):
        batch = synthetic_batch()
        actual = rollout(model, batch, source, 2, periodic)
        expected = rollout(reference, batch, source, 2, periodic)
        torch.testing.assert_close(actual, expected)
        weighted_loss(actual, batch).backward()
        weighted_loss(expected, batch).backward()
    for (name, param), (_, other) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert param.grad is not None, name
        assert other.grad is not None, name
        torch.testing.assert_close(param.grad, other.grad)


def test_rollout_never_reads_future_target_values_or_target_masks():
    model = VelocitySamudra((8, 12), checkpointing=None).eval()
    head = model.heads["duacs"]
    assert isinstance(head, torch.nn.Conv2d)
    torch.nn.init.normal_(head.weight)
    batch = synthetic_batch()
    a = rollout(model, batch, "duacs", 2)
    batch["targets"].fill_(1000)
    batch["target_valid"].fill_(False)
    b = rollout(model, batch, "duacs", 2)
    torch.testing.assert_close(a, b)


def test_geometry_encodes_latitude_dependent_spacing():
    coords, area = geometry(np.array([-60, -30, 0, 30, 60]), np.arange(8) * 45)
    np.testing.assert_allclose(np.exp(coords[3, 0] - coords[3, 2]), 0.5, rtol=1e-6)
    np.testing.assert_allclose(area[0] / area[2], 0.5, rtol=1e-6)


def test_regional_scored_interior_is_insensitive_to_larger_halo():
    torch.set_num_threads(2)
    model = VelocitySamudra((4, 6, 8, 12), checkpointing=None).eval()
    head = model.heads["om4"]
    assert isinstance(head, torch.nn.Conv2d)
    torch.nn.init.normal_(head.weight, std=0.1)
    large = synthetic_batch(640, 640, steps=1)
    small = {
        name: value[..., 128:512, 128:512] if value.ndim >= 4 else value
        for name, value in large.items()
    }
    with torch.no_grad():
        a = rollout(model, large, "om4", 1, periodic=False)
        b = rollout(model, small, "om4", 1, periodic=False)
    torch.testing.assert_close(
        a[..., 256:384, 256:384], b[..., 128:256, 128:256], rtol=1e-4, atol=1e-5
    )


def test_rust_velocity_batch_uses_training_stats_and_masks_missing_cells(tmp_path):
    rust = pytest.importorskip("samudra_rust_loader")
    t, h, w = 12, 8, 16
    data = np.broadcast_to(
        np.arange(t, dtype=np.float32)[:, None, None], (t, h, w)
    ).copy()
    data[4, 3, 3] = np.nan
    times = np.datetime64("2016-01-01") + np.arange(t) * np.timedelta64(5, "D")
    xr.Dataset(
        {"u": (("time", "y", "x"), data), "v": (("time", "y", "x"), data * 2)},
        coords={
            "time": times,
            "y": np.linspace(-70, 70, h),
            "x": np.arange(w) * 360 / w,
        },
    ).to_zarr(tmp_path / "fields.zarr", consolidated=True)
    np.savez(
        tmp_path / "stats.npz",
        mean=np.zeros((2, h, w), dtype=np.float32),
        std=np.array([1, 2], dtype=np.float32),
        mask=np.ones((h, w), dtype=bool),
        monthly_mean=np.zeros((12, 2, h, w), dtype=np.float32),
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"complete": True, "kind": "duacs"})
    )
    source = VelocitySource(tmp_path, rust.FlatOm4ReadPool(2))
    batch = source.batch(5, 2, torch.device("cpu"))
    torch.testing.assert_close(
        batch["history"][0, :, 0, 0, 0], torch.tensor([2.0, 3.0, 4.0, 5.0])
    )
    torch.testing.assert_close(
        batch["targets"][0, :, 1, 0, 0], torch.tensor([6.0, 7.0])
    )
    assert not batch["target_valid"][0, :, 0, 3, 3].any()
    assert torch.isfinite(batch["history"]).all()
