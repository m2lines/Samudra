# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Recipe integration: source exposure, cooldown, resume and replay isolation."""

import random
from collections import Counter
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from samudra.models.surface_initialized import SurfaceInitializedModel
from samudra.observations import baseline
from samudra.observations.state import TrainingState


@pytest.fixture(autouse=True)
def small_cpu_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize(
    "mix, main_counts",
    [
        ("obs", {"obs": 2000}),
        ("om4-obs", {"obs": 1000, "om4": 1000}),
        ("om4-cm4-obs", {"obs": 1000, "om4": 500, "cm4": 500}),
        ("om4-cm4", {"om4": 1000, "cm4": 1000}),
    ],
)
def test_default_recipe_allocates_exact_optimizer_updates(mix, main_counts):
    assert Counter(baseline.source_at(mix, i) for i in range(2000)) == main_counts
    assert all(baseline.source_at(mix, i) == "obs" for i in range(2000, 3000))
    assert baseline.learning_rate(2999) == pytest.approx(baseline.FINAL_LEARNING_RATE)


class TinySources:
    names = ("obs", "om4", "cm4")
    obs = SimpleNamespace(device="cpu")

    def fresh_origin(self, source, rng):
        return int(rng.integers(100))

    def layout(self, source, origin, month):
        if month > 11:
            return None
        return dict(
            start_lead=6 * month,
            end_lead=6 * (month + 1),
            next_start_lead=6 * (month + 1),
        )

    def load(self, source, origin, month=0):
        return dict(
            **self.layout(source, origin, month),
            origin=origin,
            validity=torch.ones(1, 19, 2, 1, 1, dtype=torch.bool),
        )

    def auxiliary(self, source, origin, month, sample):
        return sample


def tiny_objective(
    model, data, source, sample, auxiliary, seed, *, cached=None, coverage=None
):
    marker = 1000 * (TinySources.names.index(source) + 1)
    if cached is not None:
        # A cached state from another source or month fails at its consumer.
        assert torch.equal(
            cached, torch.full_like(cached, marker + sample["start_lead"])
        )
    target = (
        sample["origin"] / 100
        + torch.rand(()).item()
        + random.random()
        + np.random.random()
        + (seed % 100) / 100
    )
    loss = (model.weight - target).square().mean()
    next_states = torch.full((1, 2, 1, 1, 1), float(marker + sample["next_start_lead"]))
    return loss, {"forecast": loss}, next_states


def make_state():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return TrainingState(model, optimizer, TinySources.names, seed=13)


def small_recipe(monkeypatch):
    monkeypatch.setattr(baseline, "MAIN_UPDATES", 4)
    monkeypatch.setattr(baseline, "COOLDOWN_UPDATES", 2)
    monkeypatch.setattr(baseline, "VALIDATE_EVERY", 2)
    monkeypatch.setattr(baseline, "SAVE_EVERY", 1)
    monkeypatch.setattr(baseline, "objective", tiny_objective)


def validate(model):
    return {baseline.SELECTION_METRIC: float(model.weight.detach().square().mean())}


@pytest.mark.parametrize(
    "mix, main_counts",
    [
        ("om4-cm4-obs", {"obs": 2, "om4": 1, "cm4": 1}),
        ("om4-cm4", {"obs": 0, "om4": 2, "cm4": 2}),
    ],
)
def test_training_startup_populates_each_source_and_cooldown_is_obs_only(
    monkeypatch, tmp_path, mix, main_counts
):
    small_recipe(monkeypatch)
    state = make_state()
    records: list[dict[str, Any]] = []
    baseline.train(
        state,
        cast(baseline.Sources, TinySources()),
        tmp_path,
        {"mix": mix},
        validate,
        records.append,
        stop_after=4,
    )
    assert {
        name: counts["updates"] for name, counts in state.exposure.items()
    } == main_counts
    if mix == "om4-cm4":
        assert state.pools["obs"].draw(state.updates) is None
    baseline.train(
        state,
        cast(baseline.Sources, TinySources()),
        tmp_path,
        {"mix": mix},
        validate,
        records.append,
    )
    assert state.cooled and state.updates == 6
    assert (
        state.ema.num_updates == 2
    )  # Reset at cooldown, retained for both obs updates.
    assert {name: counts["updates"] for name, counts in state.exposure.items()} == {
        **main_counts,
        "obs": main_counts["obs"] + 2,
    }
    for name, counts in state.exposure.items():
        assert counts["fresh"] == counts["updates"] * 4
        assert counts["replay"] == counts["updates"] * 4
        for entries in state.pools[name].entries.values():
            assert all(entry.source == name for entry in entries)
    assert state.optimizer.state[state.model.weight]["step"] == 6
    assert (tmp_path / "main.pt").exists()
    assert (tmp_path / "endpoint.pt").exists()
    assert (tmp_path / "COMPLETE.json").exists()


@pytest.mark.parametrize("mix", ["om4-cm4-obs", "om4-cm4"])
def test_exact_resume_across_cooldown_preserves_rng_optimizer_and_replay(
    monkeypatch, tmp_path, mix
):
    small_recipe(monkeypatch)
    manifest = {"mix": mix}
    full_dir, resumed_dir = tmp_path / "full", tmp_path / "resumed"
    full_dir.mkdir()
    resumed_dir.mkdir()
    uninterrupted = make_state()
    full_records: list[dict[str, Any]] = []
    baseline.train(
        uninterrupted,
        cast(baseline.Sources, TinySources()),
        full_dir,
        manifest,
        validate,
        full_records.append,
    )
    prefix = make_state()
    resumed_records: list[dict[str, Any]] = []
    baseline.train(
        prefix,
        cast(baseline.Sources, TinySources()),
        resumed_dir,
        manifest,
        validate,
        resumed_records.append,
        stop_after=4,
    )
    assert not prefix.cooled
    restored = make_state()
    restored.restore(resumed_dir / "latest.pt", manifest)
    baseline.train(
        restored,
        cast(baseline.Sources, TinySources()),
        resumed_dir,
        manifest,
        validate,
        resumed_records.append,
    )
    torch.testing.assert_close(
        restored.model.weight, uninterrupted.model.weight, rtol=0, atol=0
    )
    assert resumed_records == full_records
    assert restored.exposure == uninterrupted.exposure
    assert restored.best_score == uninterrupted.best_score
    assert restored.ema == uninterrupted.ema
    left = restored.optimizer.state[restored.model.weight]
    right = uninterrupted.optimizer.state[uninterrupted.model.weight]
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
    for name in TinySources.names:
        for _ in range(20):
            left_entry = restored.pools[name].draw(6)
            right_entry = uninterrupted.pools[name].draw(6)
            assert (left_entry.origin, left_entry.month, left_entry.start_lead) == (
                right_entry.origin,
                right_entry.month,
                right_entry.start_lead,
            )
            torch.testing.assert_close(
                left_entry.states, right_entry.states, rtol=0, atol=0
            )


def test_replay_objective_forecast_does_not_read_auxiliary_ocean_state():
    torch.manual_seed(21)
    names = [f"{v}_{i}" for v in ("uo", "vo", "thetao", "so") for i in range(19)] + [
        "zos"
    ]
    model = SurfaceInitializedModel(names, widths=(4,), activation_checkpointing=False)
    height, width, steps = 6, 8, 2
    mask = torch.ones(77, height, width, dtype=torch.bool)
    lat = torch.linspace(-75, 75, height)
    lon = torch.arange(width) * (360 / width)
    phi, lam = torch.meshgrid(lat.deg2rad(), lon.deg2rad(), indexing="ij")
    context = torch.stack(
        (
            phi.cos() * lam.cos(),
            phi.cos() * lam.sin(),
            phi.sin(),
            torch.zeros_like(phi),
            torch.ones_like(phi),
        )
    )
    surface = torch.randn(1, 19 + steps, 2, height, width)
    sample: baseline.ForecastSample = dict(
        surface=surface,
        atmosphere=torch.randn(1, 19 + steps, 8, height, width),
        contexts=context[None, None].expand(1, 19 + steps, -1, -1, -1),
        mask=mask,
        validity=torch.ones_like(surface, dtype=torch.bool),
        month_weights=torch.ones(steps) / steps,
        start_lead=6,
        end_lead=8,
        next_start_lead=8,
    )
    captured = []

    def forecast_losses(prediction, unused):
        captured.append(prediction.detach().clone())
        error = prediction.square().mean()
        return dict(monthly_ts=error, sst=error, ssh=error)

    data: Any = SimpleNamespace(
        device="cpu",
        grid={"lat": lat},
        forecast_losses=forecast_losses,
        interior_loss=lambda prediction, sample: prediction.square().mean(),
    )
    cache = torch.randn(1, 2, 77, height, width, requires_grad=True)
    first_loss, first_parts, first_states = baseline.objective(
        model, data, "obs", sample, sample, 5, cached=cache
    )
    modified: baseline.ForecastSample = {**sample, "surface": sample["surface"] + 100}
    second_loss, second_parts, second_states = baseline.objective(
        model, data, "obs", sample, modified, 5, cached=cache
    )
    torch.testing.assert_close(captured[0], captured[1], rtol=0, atol=0)
    torch.testing.assert_close(first_states, second_states, rtol=0, atol=0)
    assert first_parts["reconstruction"] != second_parts["reconstruction"]
    second_loss.backward()
    assert cache.grad is None
    assert torch.isfinite(first_loss) and torch.isfinite(second_loss)
    assert not second_states.requires_grad


def test_annual_driver_initializes_once_and_never_reads_future_ocean():
    from samudra.observations.evaluate import complete_month_weights

    starts = np.datetime64("2013-11-01", "ns") + np.arange(73) * np.timedelta64(5, "D")
    ends = starts + np.timedelta64(5, "D")
    months, matrix = complete_month_weights(starts, ends)
    surface = torch.full((1, 92, 2, 1, 1), float("nan"))
    surface[:, :19] = 0
    trajectory = dict(
        interval_starts=starts,
        interval_ends=ends,
        months=months,
        month_matrix=matrix,
        raw_surface=torch.zeros(1, 73, 2, 1, 1),
        monthly_targets=torch.zeros(1, 12, 28, 1, 1),
        surface=surface,
        atmosphere=torch.zeros(1, 92, 8, 1, 1),
        contexts=torch.zeros(1, 92, 5, 1, 1),
        validity=torch.isfinite(surface),
    )
    profile_mask = torch.ones(28, 1, 1, dtype=torch.bool)
    profile_mask[0] = False
    data: Any = SimpleNamespace(
        std=torch.ones(77),
        area=torch.ones(1, 1),
        mask=torch.ones(77, 1, 1, dtype=torch.bool),
        ts_mask=profile_mask,
        device="cpu",
        physical=lambda value: value,
    )

    class ContinuousModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.initializations = 0
            self.steps = 0

        def initialize(self, surface, atmosphere, context, mask, validity):
            self.initializations += 1
            assert surface.shape[1] == 19 and torch.isfinite(surface).all()
            return torch.zeros(1, 2, 77, 1, 1)

        def continue_forecast(self, states, atmosphere, context, mask, start_lead):
            assert start_lead == self.steps + 1
            assert torch.all(states[:, -1] == self.steps)
            assert atmosphere.shape[1] == context.shape[1] == 1
            self.steps += 1
            predicted = states[:, -1:] + 1
            return predicted, torch.cat((states[:, -1:], predicted), dim=1)

    model: Any = ContinuousModel()
    logs = baseline.annual_metrics(
        model, data, cast(baseline.ForecastSample, trajectory)
    )
    assert model.initializations == 1 and model.steps == 73
    assert logs["rollout_val/365d/weighted_rmse/thetao_0"] == 37
    assert np.isfinite(logs[baseline.SELECTION_METRIC])


def test_shared_autocast_honors_explicit_cpu_device(monkeypatch):
    from samudra.utils import device

    monkeypatch.setattr(device, "_CHOSEN_DEVICE", torch.device("cuda"))
    values = torch.ones(2, 2)
    with device.autocast(True, torch.bfloat16, device="cpu"):
        assert (values @ values).dtype == torch.bfloat16
    with device.autocast(False, torch.bfloat16, device="cpu"):
        assert (values @ values).dtype == torch.float32
