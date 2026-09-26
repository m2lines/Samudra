# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import datetime
from types import SimpleNamespace

import cftime
import numpy as np
import pytest
import torch

from samudra.experiments.frame_cache import PreparedFrameCache
from samudra.experiments.initializer_models import (
    AttentionBlock,
    HistoryInitializer,
    SwinReconstructor,
)
from samudra.experiments.initializer_wave import InitializerWave
from samudra.experiments.surface_state import geographic_features

NAMES = ["uo_0", "vo_0", "thetao_0", "thetao_1", "so_0", "zos"]


def test_short_inputs_exclude_old_surfaces_and_all_forcing():
    torch.set_num_threads(1)
    model = HistoryInitializer(NAMES, "unet", False)
    model.net = torch.nn.Conv2d(29, 2 * len(NAMES), 1)
    surface = torch.randn(2, 19, 2, 16, 32)
    past = torch.randn(2, 19, 3, 16, 32)
    context = torch.randn(2, 5, 16, 32)
    mask = torch.ones(len(NAMES), 16, 32, dtype=torch.bool)
    mask[:, :2] = False
    expected = model(surface, past, context, mask)
    altered = surface.clone()
    altered[:, :13] += 1000
    actual = model(altered, past + 1000, context, mask)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual[:, :, [2, 5]], surface[:, -2:] * mask[[2, 5]])
    assert not actual[..., :2, :].any()


def test_expanded_inputs_use_old_forcing_and_surfaces():
    model = HistoryInitializer(NAMES, "unet", True)
    model.net = torch.nn.Conv2d(138, 2 * len(NAMES), 1)
    surface = torch.randn(1, 19, 2, 8, 16, requires_grad=True)
    past = torch.randn(1, 19, 3, 8, 16, requires_grad=True)
    prediction = model(
        surface,
        past,
        torch.randn(1, 5, 8, 16),
        torch.ones(len(NAMES), 8, 16, dtype=torch.bool),
    )
    prediction[:, :, 3].square().mean().backward()
    assert surface.grad is not None and past.grad is not None
    assert surface.grad[:, 0].abs().sum() > 0
    assert past.grad[:, 0].abs().sum() > 0


def test_shifted_attention_wraps_longitude_but_not_latitude():
    torch.manual_seed(3)
    block = AttentionBlock(32, 1, window=6, shift=3).eval()
    x = torch.randn(1, 12, 12, 32)
    expected = block(x)
    changed = x.clone()
    changed[:, 0] += torch.randn_like(changed[:, 0]) * 10
    actual = block(changed)
    torch.testing.assert_close(actual[:, -1], expected[:, -1])
    changed = x.clone()
    changed[:, :, 0] += torch.randn_like(changed[:, :, 0]) * 10
    actual = block(changed)
    assert (actual[:, :, -1] - expected[:, :, -1]).abs().max() > 0.001


def test_swin_odd_shapes_and_global_attention_gradient():
    model = SwinReconstructor(5, 12, widths=(32, 64, 96, 128), depths=(2, 2, 2, 2))
    x = torch.randn(1, 35, 49, 5).permute(0, 3, 1, 2).requires_grad_()
    out = model(x)
    assert out.shape == (1, 12, 35, 49)
    out.square().mean().backward()
    assert torch.isfinite(out).all()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    stage = model.stages[-1]
    assert isinstance(stage, torch.nn.ModuleList)
    block = stage[0]
    assert isinstance(block, AttentionBlock)
    assert block.qkv.weight.grad is not None
    assert block.qkv.weight.grad.abs().sum() > 0
    detail = model.detail[0]
    assert isinstance(detail, torch.nn.Conv2d)
    assert detail.weight.grad is not None
    assert detail.weight.grad.abs().sum() > 0


def test_compact_sampling_uses_past_only_and_correct_forecast_alignment(
    monkeypatch: pytest.MonkeyPatch,
):
    experiment = InitializerWave.__new__(InitializerWave)
    experiment.device = torch.device("cpu")
    experiment.initializer = SimpleNamespace(surface=[2, 5])
    experiment.mask = torch.ones(6, 2, 4, dtype=torch.bool)
    experiment.geo = geographic_features(
        torch.tensor([-30.0, 30.0]), torch.arange(4).float()
    )
    dates = np.array(
        [
            cftime.DatetimeNoLeap(2000, 1, 1) + datetime.timedelta(days=5 * i)
            for i in range(40)
        ]
    )
    source = SimpleNamespace(time=SimpleNamespace(values=dates))
    dataset = SimpleNamespace(sources=[source], steps=6)
    cache = PreparedFrameCache(40, 6, 3, (2, 4), torch.device("cpu"))
    cache.prognostic.copy_(torch.arange(40.0).view(40, 1, 1, 1))
    cache.boundary.copy_((100 + torch.arange(40.0)).view(40, 1, 1, 1))
    experiment.frame_caches = {id(source): cache}
    surface, past, context, truth, forcing, labels = experiment.sample(dataset, [0, 3])
    torch.testing.assert_close(surface[0, :, 0, 0, 0], torch.arange(19.0))
    torch.testing.assert_close(past[0, :, 0, 0, 0], 100 + torch.arange(19.0))
    torch.testing.assert_close(truth[0, :, 0, 0, 0], torch.tensor([17.0, 18.0]))
    torch.testing.assert_close(forcing[0, :, 0, 0, 0], 100 + torch.arange(18.0, 24.0))
    torch.testing.assert_close(labels[0, :, 0, 0, 0], torch.arange(19.0, 25.0))
    torch.testing.assert_close(labels[1, :, 0, 0, 0], torch.arange(22.0, 28.0))

    class NativeBatch:
        def __len__(self):
            return 6

        def get_input(self, step):
            # Native batches retain all 19 history frames at each forecast step.
            times = torch.tensor([0, 3])[:, None] + torch.arange(step, step + 19)
            return (
                cache.prognostic[times].flatten(1, 2),
                cache.boundary[times].flatten(1, 2),
            )

        def get_initial_input(self):
            return self.get_input(0)

        def get_label(self, step):
            return cache.prognostic[torch.tensor([19, 22]) + step]

    experiment.channels = 6
    monkeypatch.setattr(
        experiment, "native_loader", lambda dataset, sampler: iter([NativeBatch()])
    )
    expected = (surface, past, context, truth, forcing, labels)
    experiment.frame_caches.clear()
    streamed = experiment.sample(dataset, [0, 3])
    for actual, wanted in zip(streamed, expected, strict=True):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)


def test_long_context_preserves_target_period_configuration():
    config = InitializerWave.__new__(InitializerWave).build_data_config(
        SimpleNamespace(readers=8)
    )
    assert config.input_steps == 19
    assert str(config.sources[0].train_time.end) == "2013-10-04"
    assert str(config.sources[0].val_time.start) == "2013-10-05"
    assert str(config.sources[0].inference_times[0].start) == "2014-10-10"

    experiment = InitializerWave.__new__(InitializerWave)
    experiment.config = config
    context = experiment.context_config()
    assert str(context.sources[0].val_time.start) == "2013-08-01"
    assert str(context.sources[0].inference_times[0].start) == "2014-08-06"
    assert str(experiment.config.sources[0].train_time.end) == "2013-10-04"


def test_streaming_reconstruction_decomposition_and_fixed_climatology(tmp_path):
    import pandas as pd

    from samudra.experiments.initializer_diagnostics import ReconstructionDiagnostics

    names = [f"{v}_{i}" for v in ["thetao", "so", "uo", "vo"] for i in range(19)] + [
        "zos"
    ]
    root = tmp_path / "models"
    (root / "initializer").mkdir(parents=True)
    torch.save(
        {"monthly": torch.zeros(12, 77, 6, 12)}, root / "initializer" / "climatology.pt"
    )
    out = tmp_path / "diagnostics"
    out.mkdir()
    lat = torch.tensor([-45.0, -25.0, -5.0, 5.0, 25.0, 45.0])
    e = SimpleNamespace(
        device=torch.device("cpu"),
        mask=torch.ones(77, 6, 12, dtype=torch.bool),
        lat=lat,
        weights=torch.ones(77, 6, 12),
        channels=77,
        pretrain=root / "ar" / "pretrain-best.pt",
        names=names,
        source=SimpleNamespace(resolution=(lat, torch.arange(12))),
        bundle=SimpleNamespace(
            data_layout=SimpleNamespace(depth_levels=list(range(19)))
        ),
        rank=0,
        out=out,
        reduce=lambda tensor: tensor,
    )
    source = SimpleNamespace(
        time=SimpleNamespace(
            values=np.array(
                [
                    cftime.DatetimeNoLeap(2000, 1, 1) + datetime.timedelta(days=i)
                    for i in range(20)
                ]
            )
        )
    )
    diagnostics = ReconstructionDiagnostics(e, source, [0, 1])
    truth = torch.randn(2, 77, 6, 12)
    diagnostics.add(truth + 2, truth, [0, 1])
    diagnostics.finish()
    table = pd.read_csv(out / "temporal-decomposition.csv")
    np.testing.assert_allclose(table["mse"], 4, atol=1e-6)
    np.testing.assert_allclose(table["mean_bias_mse"], 4, atol=1e-6)
    np.testing.assert_allclose(table["temporal_amplitude_mse"], 0, atol=1e-10)
    np.testing.assert_allclose(table["temporal_pattern_mse"], 0, atol=1e-10)
    scales = pd.read_csv(out / "spatial-scales.csv")
    np.testing.assert_allclose(
        scales.loc[scales.box_width_cells > 0, "anomaly_mse"], 0, atol=1e-10
    )
    snapshots = np.load(out / "snapshots-rank0.npz")
    assert {"0_prediction", "1_truth"} <= set(snapshots.files)


def test_initial_only_reader_requests_surface_history_and_aligned_state_pair(
    monkeypatch,
):
    experiment = InitializerWave.__new__(InitializerWave)
    experiment.device = torch.device("cpu")
    experiment.initializer = SimpleNamespace(surface=[2, 5])
    experiment.names, experiment.channels = NAMES, 6
    experiment.mask = torch.ones(6, 2, 4)
    experiment.geo = geographic_features(
        torch.tensor([-30.0, 30.0]), torch.arange(4).float()
    )
    experiment.bundle = SimpleNamespace(
        data_layout=SimpleNamespace(boundary_var_names=["a", "b", "c"])
    )
    experiment.frame_caches, experiment.initial_views = {}, {}
    dates = np.array(
        [
            cftime.DatetimeNoLeap(2000, 1, 1) + datetime.timedelta(days=5 * i)
            for i in range(40)
        ]
    )
    source = SimpleNamespace(time=SimpleNamespace(values=dates))
    dataset = SimpleNamespace(sources=[source])
    calls = []

    def constructor(**kwargs):
        return SimpleNamespace(**kwargs)

    def loader(view, sampler):
        ids = sampler[0]
        calls.append((view.prognostic_var_names, view.input_steps, ids))
        channels = torch.tensor([NAMES.index(n) for n in view.prognostic_var_names])
        times = torch.tensor(ids)[:, None] + torch.arange(view.input_steps)
        values = (
            (times[:, :, None] * 10 + channels)
            .float()[..., None, None]
            .expand(-1, -1, -1, 2, 4)
        )
        boundary = (times + 100).float()[:, :, None, None, None].expand(-1, -1, 3, 2, 4)
        return iter(
            [
                SimpleNamespace(
                    get_initial_input=lambda: (
                        values.flatten(1, 2),
                        boundary.flatten(1, 2),
                    )
                )
            ]
        )

    monkeypatch.setattr(
        "samudra.experiments.initializer_wave.TorchTrainDataset", constructor
    )
    monkeypatch.setattr(experiment, "native_loader", loader)
    surface, past, _, truth = experiment.initial_sample(dataset, [0, 3])
    assert calls == [(["thetao_0", "zos"], 19, [0, 3]), (NAMES, 2, [17, 20])]
    torch.testing.assert_close(surface[0, :, 0, 0, 0], torch.arange(19.0) * 10 + 2)
    torch.testing.assert_close(past[1, :, 0, 0, 0], torch.arange(3.0, 22.0) + 100)
    torch.testing.assert_close(truth[0, :, 0, 0, 0], torch.tensor([170.0, 180.0]))
    torch.testing.assert_close(truth[1, :, 5, 0, 0], torch.tensor([205.0, 215.0]))
