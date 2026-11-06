# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Streaming must preserve the original sampled recurrent path exactly."""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from samudra.experiments.samudra_stochastic_transition import (  # type: ignore[import-not-found]
    SamudraStochasticTransition,
)


def fixture(arm="C", **kwargs):
    torch.set_num_threads(1)
    torch.manual_seed(19)
    model = SamudraStochasticTransition(
        SimpleNamespace(channels=4, history=3, surface=[0, 3]),
        arm=arm,
        widths=(4, 8),
        **kwargs,
    )
    surface = torch.randn(1, 5, 2, 17, 25)
    atmosphere = torch.randn(1, 5, 8, 17, 25)
    contexts = torch.randn(1, 5, 5, 17, 25)
    mask = torch.ones(4, 17, 25)
    mask[:, 0] = 0
    validity = torch.ones_like(surface)
    validity[:, 1, 0, 4, 4] = 0
    return model, (surface, atmosphere, contexts, mask, validity)


def rng():
    return torch.Generator().manual_seed(7)


def activate_noise(model):
    with torch.no_grad():
        for net in (model.initializer, model.processor):
            for group in net.film.values():
                for head in group:
                    head.global_head[-1].weight.fill_(0.03)
                    head.spatial_head.weight.fill_(0.03)


spec = importlib.util.spec_from_file_location(
    "annual_export",
    Path(os.environ["SIGNED_EXPORTER"]),
)
assert spec is not None and spec.loader is not None
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


@pytest.mark.parametrize("arm", ["B"])
def test_stream_is_exact_forecast_and_rng(arm):
    model, args = fixture(arm, forcing_mode="direct")
    model.eval()
    activate_noise(model)
    original_rng = torch.Generator().manual_seed(904003)
    stream_rng = torch.Generator().manual_seed(904003)
    with torch.no_grad():
        expected, initial = model.forecast(*args, generator=original_rng, members=8)
        actual = list(export.stream_forecast(model, args, stream_rng, members=8))
    assert [lead for lead, _ in actual] == [0, 5, 10]
    torch.testing.assert_close(actual[0][1], initial[:, :, -1], rtol=0, atol=0)
    for index, (_, field) in enumerate(actual[1:]):
        torch.testing.assert_close(field, expected[:, :, index], rtol=0, atol=0)
    assert torch.equal(stream_rng.get_state(), original_rng.get_state())


def test_stream_short_prefix_unchanged_by_later_forcing():
    model, args = fixture("B", forcing_mode="direct")
    activate_noise(model)
    longer = list(args)
    longer[1] = torch.cat((longer[1], longer[1][:, -2:]), 1)
    longer[2] = torch.cat((longer[2], longer[2][:, -2:]), 1)
    with torch.no_grad():
        short = list(
            export.stream_forecast(model, args, torch.Generator().manual_seed(2))
        )
        long = list(
            export.stream_forecast(model, longer, torch.Generator().manual_seed(2))
        )
    for (_, a), (_, b) in zip(short, long):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert len(long) == len(short) + 2


def test_nonfinite_preserves_prefix_and_marks_months_incomplete(tmp_path):
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd

    class FailingModel:
        history, channels = 19, 77

        def encode_native(self, surface, atmosphere, context, mask, validity, **kwargs):
            return torch.zeros(1, 1, 2, 3), None, None

        def initialize(self, conditioning, known, anchor, mask, generator):
            return torch.zeros(len(conditioning), 2, 77, 2, 3)

        def advance(self, states, forcing, context, mask, generator):
            following = states[:, -1] + 1
            following = torch.where(following >= 3, float("nan"), following)
            return torch.stack((states[:, -1], following), 1)

        def forecast(self, *arguments, generator, members):
            frames = list(export.stream_forecast(self, arguments, generator, members))
            return torch.stack([s for _, s in frames[1:]], 2), frames[0][1][
                :, :, None
            ].expand(-1, -1, 2, -1, -1, -1)

    names = np.asarray(
        [f"{v}_{i}" for v in ("uo", "vo", "thetao", "so") for i in range(19)] + ["zos"]
    )
    grid = dict(
        names=names,
        mask=np.ones((77, 2, 3), dtype=bool),
        mean=np.zeros(77, dtype=np.float32),
        std=np.ones(77, dtype=np.float32),
        lat=np.array([-30, 30]),
        lon=np.array([0, 120, 240]),
    )
    data = SimpleNamespace(grid=grid, device="cpu")
    raw = dict(
        start=pd.date_range("2014-09-28", periods=92, freq="5D").astype(str).to_numpy()
    )
    arguments = (
        torch.zeros(1, 19, 2, 2, 3),
        torch.zeros(1, 92, 8, 2, 3),
        torch.zeros(1, 92, 5, 2, 3),
        torch.ones(77, 2, 3),
        torch.ones(1, 19, 2, 2, 3),
    )
    result = export.run_origin(
        FailingModel(),
        data,
        arguments,
        raw,
        tmp_path,
        7,
        [f"2015-{i:02d}" for i in range(1, 13)],
        label="obs",
    )
    assert result["completed_bins"] == 2 and not result["complete"]
    assert result["first_nonfinite"]["lead"] == 15
    assert (tmp_path / "NONFINITE.npz").exists()
    with np.load(tmp_path / "mean8.npz") as saved:
        assert saved["surface"].shape[0] == 2
        assert saved["interior_leads"].tolist() == [5]
        assert np.isnan(saved["monthly_ts"]).all()
        assert saved["leads"].tolist() == [5, 10]
