# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Any

import cftime
import numpy as np
import pytest
import torch
import xarray as xr

from samudra.skyline.data import CHANNELS, OceanSource
from samudra.skyline.model import ModelConfig, PointTransformer, Vocabulary
from samudra.skyline.run import FrameStream


@pytest.fixture
def ocean(tmp_path):
    dates = [
        cftime.DatetimeJulian(y, m, d)
        for y, m, days in [
            (1975, 1, [3, 8, 13]),
            (2013, 10, [5, 10, 15]),
            (2014, 10, [10, 15, 20]),
        ]
        for d in days
    ]
    a = np.arange(9 * 5 * 8, dtype=np.float32).reshape(9, 5, 8) / 100
    fields: dict[str, Any] = {
        n: (("time", "y", "x"), a + i) for i, n in enumerate(CHANNELS)
    }
    wet = np.ones((5, 8), dtype=bool)
    wet[2, 3] = False
    fields.update({f"mask_{z}": (("y", "x"), wet) for z in range(19)})
    fields["areacello"] = (("y", "x"), np.ones((5, 8)))
    xr.Dataset(
        fields,
        coords={"time": dates, "y": np.linspace(-80, 80, 5), "x": np.arange(8) * 45},
    ).to_zarr(tmp_path / "OM4.zarr", consolidated=True)
    for name, value in [("means", 1.0), ("stds", 2.0)]:
        xr.Dataset({n: xr.DataArray(value) for n in CHANNELS}).to_zarr(
            tmp_path / f"OM4_{name}.zarr", consolidated=True
        )
    source = OceanSource(str(tmp_path), 2)
    yield source
    source.close()


def test_split_and_cadence(ocean):
    assert {k: v.tolist() for k, v in ocean.origins.items()} == {
        "train": [1],
        "validation": [4],
        "test": [7],
    }


def test_crop_wrap_padding_masks_and_target_isolation(ocean):
    ocean.locations = [np.array([0])] * 19
    ocean.probabilities = [np.array([1.0])] * 19
    frame = ocean.read(1)
    batch = ocean.sample(frame, np.random.default_rng(2), 2, 3, [0, 18])
    assert batch["values"].shape == (2, 18, 80)
    assert not batch["valid"][:, :3].any()
    assert not batch["valid"][:, 9:12].any()
    assert (batch["values"][:, :3] == 0).all()
    assert batch["values"][0, 3, 0] == frame["input"][0, 0, 0, 7]
    assert batch["current"][0] == frame["input"][1, 38, 0, 0]
    assert batch["target"][0] == frame["target"][0, 0, 0]
    changed = dict(frame, target=frame["target"] + 10)
    other = ocean.sample(changed, np.random.default_rng(2), 2, 3, [0, 18])
    for key in batch:
        if key != "target":
            torch.testing.assert_close(batch[key], other[key])
    torch.testing.assert_close(other["target"], batch["target"] + 10)


def test_histogram_density_and_midpoint_assignment():
    vocab = Vocabulary(np.full(19, 10.0), 17)
    z = torch.tensor([0, 0])
    mid = vocab.edges[0, 5]
    assert vocab.encode(torch.tensor([mid - 1e-4, mid + 1e-4]), z).tolist() == [4, 5]
    depths = torch.zeros(17, dtype=torch.long)
    logits = torch.arange(17, dtype=torch.float32).repeat(17, 1) / 5
    std = torch.full((19,), 3.0)
    density = vocab.log_density(logits, vocab.centers[0], depths, std).exp()
    widths = vocab.edges[0].diff() * std[0]
    torch.testing.assert_close((density * widths).sum(), torch.tensor(1.0))
    outside = vocab.log_density(
        torch.zeros(1, 17), torch.tensor([100.0]), torch.tensor([0]), std
    )
    assert torch.isneginf(outside).all()


@pytest.mark.parametrize("objective", ["absolute_ce", "residual_ce", "regression"])
def test_forward_backward_and_masked_context(ocean, objective):
    torch.set_num_threads(2)
    batch = ocean.sample(ocean.read(1), np.random.default_rng(4), 2, 3)
    model = PointTransformer(
        ModelConfig(width=32, layers=2, heads=4, bins=17, objective=objective)
    )
    logits = model(batch)
    assert logits.shape == (2, 1 if objective == "regression" else 17)
    logits.square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    model.eval()
    altered = {k: v.clone() for k, v in batch.items()}
    altered["target"] += 1e4
    altered["values"][~batch["valid"]] = 1000
    with torch.no_grad():
        torch.testing.assert_close(model(batch), model(altered))


def test_frame_stream_resume():
    class Source:
        origins = {"train": np.array([1, 2, 3, 4])}

        def read(self, t):
            return {"origin": t}

    a = FrameStream(Source(), 42)
    a.advance()
    a.used = 8
    b = FrameStream(Source(), 42, copy.deepcopy(a.state()))
    assert a.state() == b.state()
    for _ in range(12):
        a.advance()
        b.advance()
        assert a.frame == b.frame
        assert a.state() == b.state()
    a.close()
    b.close()
