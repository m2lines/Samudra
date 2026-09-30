# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from samudra.config import BlockConfig, UNetBackboneConfig
from samudra.models.surface_initialized import SurfaceInitializedConfig
from samudra.observations import metrics
from samudra.observations.archive import month_intervals, strict_mean
from samudra.observations.losses import corrupt_sample
from samudra.observations.prepare import ConservativeRemap, interpolate_depth
from samudra.tasks.observation import Harness, flat_metrics, sample_indices, task_counts


def small_model():
    backbone = UNetBackboneConfig(
        ch_width=[8, 12],
        dilation=[1, 1],
        n_layers=[1, 1],
        core_block=BlockConfig(upscale_factor=2, norm="instance", instance_affine=True),
    )
    return SurfaceInitializedConfig(initializer=backbone, processor=backbone).build(
        ["thetao_0", "zos", "thetao_1"]
    )


def test_observed_only_copy_and_learned_gap_gradients():
    model = small_model().train()
    surface = torch.randn(1, 19, 2, 8, 16)
    visible = torch.ones_like(surface)
    visible[:, :, :, :, 4:8] = 0
    supplied = surface * visible
    mask = torch.ones(3, 8, 16)
    result = model.initialize(
        supplied, torch.randn(1, 19, 8, 8, 16), torch.randn(1, 5, 8, 16), mask, visible
    )
    torch.testing.assert_close(
        result[:, :, :2][visible[:, -2:].bool()],
        supplied[:, -2:][visible[:, -2:].bool()],
        rtol=0,
        atol=0,
    )
    result[:, :, :2][~visible[:, -2:].bool()].square().mean().backward()
    assert model.initializer.net[-1].weight.grad.abs().sum() > 0
    assert all(
        m.affine and not m.track_running_stats
        for m in model.modules()
        if isinstance(m, nn.InstanceNorm2d)
    )
    assert not any(isinstance(m, nn.BatchNorm2d) for m in model.modules())


def test_future_surfaces_never_enter_forecast():
    model = small_model().eval()
    mask = torch.ones(3, 8, 16)
    surface = torch.randn(1, 21, 2, 8, 16)
    valid = torch.ones_like(surface)
    atmosphere = torch.randn(1, 21, 8, 8, 16)
    contexts = torch.randn(1, 21, 5, 8, 16)
    with torch.no_grad():
        expected = model.forecast(surface, atmosphere, contexts, mask, valid)[0]
        surface[:, 19:] = 1e5
        valid[:, 19:] = 0
        actual = model.forecast(surface, atmosphere, contexts, mask, valid)[0]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_gaps_do_not_change_targets_or_global_rng():
    sample = {
        "surface": torch.ones(1, 25, 2, 16, 32),
        "validity": torch.ones(1, 25, 2, 16, 32),
    }
    state = torch.get_rng_state()
    corrupted = corrupt_sample(sample, 1729)
    assert torch.equal(state, torch.get_rng_state())
    assert sample["surface"].min() == 1
    assert corrupted["completion_valid"].any()
    assert corrupted["completion_target"].min() == 1
    assert not corrupted["surface"][~corrupted["validity"]].any()


def test_calendar_weights_and_strict_temporal_support():
    starts, ends, weights = month_intervals("2000-02")
    assert len(starts) == 25 and weights[-1] == pytest.approx(4 / 29)
    assert starts[0] == pd.Timestamp("2000-02-01") - pd.Timedelta(days=95)
    assert ends[-1] == pd.Timestamp("2000-03-02")
    assert weights.sum() == pytest.approx(1)
    result = strict_mean(np.array([[1.0, 2.0], [3.0, np.nan]]))
    assert result[0] == 2 and np.isnan(result[1])


def test_conservative_identity_and_missing_depth_brackets():
    remap = ConservativeRemap([-45, 45], [90, 270], [-45, 45], [90, 270])
    values = np.array([[1.0, np.nan], [2.0, 3.0]])
    actual, coverage = remap(values)
    np.testing.assert_array_equal(actual, values)
    assert coverage[0, 1] == 0
    z = np.array([0.0, 10.0])
    v = np.array([[[2.0]], [[np.nan]]])
    out = interpolate_depth(v, z, target_depth=np.array([0.0, 5.0, 20.0]))
    assert out[0, 0, 0] == 2 and np.isnan(out[1:]).all()


def test_polar_errors_are_scored(monkeypatch):
    monkeypatch.setattr(metrics, "spatial_error", lambda *a, **k: None)
    lat = np.linspace(-85, 85, 18)
    lon = np.arange(36) * 10
    rng = np.random.default_rng(1)
    pred = rng.normal(size=(3, 6, 2, 18, 36)) * 0.01
    ref = np.concatenate([pred, rng.normal(size=pred.shape) * 0.01], axis=2)
    args = dict(
        reference=ref,
        predicted_ohc=np.ones((3, 2, 18, 36)),
        reference_ohc=np.ones((3, 2, 18, 36)),
        lat=lat,
        lon=lon,
        mask=np.ones((18, 36), bool),
    )
    original = metrics.score(pred, **args)
    pred[:, :, 0, lat > 60] += 3
    changed = metrics.score(pred, **args)
    assert original["metrics"]["sst_rmse"] == 0
    assert changed["metrics"]["sst_rmse"] > 0


def test_missing_adt_stencil_does_not_affect_velocity(monkeypatch):
    monkeypatch.setattr(metrics, "spatial_error", lambda *a, **k: None)
    lat, lon = np.linspace(-55, 55, 12), np.arange(16) * 22.5
    rng = np.random.default_rng(4)
    p = rng.normal(size=(3, 6, 2, 12, 16)) * 0.01
    r = rng.normal(size=(3, 6, 4, 12, 16)) * 0.01
    r[:, :, 1, 7, 8] = np.nan
    changed = p.copy()
    changed[:, :, 1, 7, 8] += np.arange(3)[:, None] * 100
    args = dict(
        reference=r,
        predicted_ohc=np.zeros((3, 2, 12, 16)),
        reference_ohc=np.ones((3, 2, 12, 16)),
        lat=lat,
        lon=lon,
        mask=np.ones((12, 16), bool),
    )
    a, b = metrics.score(p, **args), metrics.score(changed, **args)
    for k in ["velocity_rmse", "eke_rmse"]:
        assert a["metrics"][k] == pytest.approx(b["metrics"][k])


def test_plan_and_sampling_resume():
    assert task_counts(16000) == {"om4": 8000, "observation": 8000}
    previous = task_counts(0)
    for i in range(1, 16001):
        current = task_counts(i)
        assert sorted(current[k] - previous[k] for k in current) == [0, 1]
        previous = current
    whole = [sample_indices(13, 1729, i) for i in range(40)]
    assert whole[17:] == [sample_indices(13, 1729, i) for i in range(17, 40)]


def test_nested_metrics_are_logged_with_distinct_names():
    actual = flat_metrics(
        {
            "validation": {
                "metrics": {"sst_rmse": 1},
                "spectra": {
                    "sst/region/day30": {"error_dex": 0.2, "prediction_power": [1, 2]}
                },
            },
            "om4_retention": {"ts_mse": 0.4},
        }
    )
    assert actual == {
        "validation/metrics/sst_rmse": 1.0,
        "validation/spectra/sst/region/day30/error_dex": 0.2,
        "om4_retention/ts_mse": 0.4,
    }
    with pytest.raises(ValueError):
        flat_metrics({"loss": float("nan")})


def test_checkpoint_restores_optimizer_and_rng(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_rng_state", torch.get_rng_state)
    monkeypatch.setattr(torch.cuda, "set_rng_state", torch.set_rng_state)
    h = Harness.__new__(Harness)
    h.device = torch.device("cpu")
    h.model = nn.Linear(3, 2)
    h.optimizer = torch.optim.AdamW(h.model.parameters(), lr=1e-4)
    h.manifest = {"version": "fixture"}
    h.completed = 8
    h.best = 0.5

    def update():
        h.optimizer.zero_grad()
        loss = h.model(torch.randn(4, 3)).square().mean()
        loss.backward()
        h.optimizer.step()

    update()
    h.checkpoint(tmp_path / "last.pt")
    update()
    expected = {k: v.clone() for k, v in h.model.state_dict().items()}
    h.restore(tmp_path / "last.pt")
    update()
    for k, v in h.model.state_dict().items():
        torch.testing.assert_close(v, expected[k], rtol=0, atol=0)
    h.manifest = {"version": "changed"}
    with pytest.raises(ValueError, match="contract"):
        h.restore(tmp_path / "last.pt")


def test_annual_inputs_discard_future_surface_values():
    from types import SimpleNamespace

    from samudra.observations.annual import inputs

    lat = np.linspace(-85, 85, 8)
    lon = np.arange(16) * 22.5
    data = SimpleNamespace(
        grid={
            "mask": np.ones((77, 8, 16), bool),
            "lat": lat,
            "lon": lon,
            "mean": np.zeros(77),
            "std": np.ones(77),
        },
        stats={
            "surface_climatology": np.zeros((12, 2, 8, 16)),
            "atmosphere_mean": np.zeros(8),
            "atmosphere_std": np.ones(8),
        },
        mask=torch.ones(77, 8, 16),
        tensor=lambda x: torch.as_tensor(x, dtype=torch.float32),
    )
    raw: dict[str, Any] = {
        "surface": np.ones((92, 2, 8, 16)),
        "atmosphere": np.ones((92, 8, 8, 16)),
        "midpoint": pd.date_range("2014-09-28", periods=92, freq="5D"),
    }
    raw["surface"][0, 0, 0, 0] = np.nan
    before = inputs(data, raw)
    raw["surface"][19:] = 1e6
    after = inputs(data, raw)
    for a, b in zip(before, after, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert before[0].shape[1] == 19
    assert before[0][0, 0, 0, 0, 0] == 0 and before[-1][0, 0, 0, 0, 0] == 0
    assert before[1].shape[1] == 92


def test_harness_initializes_mainline_logger_and_typed_components(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from samudra.observations.archive import SPLITS
    from samudra.tasks import observation
    from samudra.utils.multiton import MultitonScope

    class Data:
        grid = {"names": np.array(["thetao_0", "zos", "thetao_1"])}

        def use_observation_normalization(self):
            pass

        def paths(self, split):
            return [
                tmp_path / f"{m}.npz" for m in pd.period_range(*SPLITS[split], freq="M")
            ]

    device = torch.device("cpu")
    calls = []

    def select_device(*args):
        calls.append(args)
        return device

    monkeypatch.setattr(observation.torch, "device", select_device)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(observation, "Samples", lambda *_: Data())
    model = tmp_path / "model.yaml"
    model.write_text(
        "initializer: &net\n  ch_width: [8, 12]\n  dilation: [1, 1]\n  n_layers: [1, 1]\n  core_block: {norm: instance, instance_affine: true}\nprocessor: *net\n"
    )
    args = SimpleNamespace(
        output=tmp_path / "run",
        observations=tmp_path / "data",
        om4=tmp_path / "om4",
        model=str(model),
        om4_config="observation/om4.yaml",
        command="evaluate",
        wandb_mode="disabled",
        entity=None,
        project="test",
        name="test",
    )
    with MultitonScope():
        harness = Harness(args)
        assert ("cuda", 0) in calls
        assert harness.model.initializer.channels == 3
        assert not harness.log.enabled
        harness.emit("fixture", {"validation": {"metrics": {"sst_rmse": 1.0}}})
        assert (args.output / "events.jsonl").exists()


def test_already_reached_partial_limit_does_not_train_again(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from samudra.tasks import observation

    harness = Harness.__new__(Harness)
    harness.args = SimpleNamespace(stop_after=8)
    monkeypatch.setattr(harness, "om4", object(), raising=False)
    harness.out = tmp_path
    harness.completed = 8
    harness.stop = False
    monkeypatch.setattr(harness, "reference", lambda: None)
    saved = []
    monkeypatch.setattr(
        harness, "checkpoint", lambda path: saved.append(harness.completed)
    )
    monkeypatch.setattr(observation.signal, "signal", lambda *args: None)
    harness.train()
    assert saved == [8]
