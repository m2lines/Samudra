# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Scientific contracts: calendar alignment, held-out exclusion and gradient paths."""

import copy
import io
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from samudra.experiments.monthly_pilot_data import observation_loss
from samudra.experiments.monthly_replay import ReplayPool, replay_loss, window_layout


def test_calendar_replay_keeps_partial_bins_and_absolute_leads():
    lo, hi, starts, weights, duration, position = window_layout("2001-05", "2001-06")
    assert (lo, hi) == (6, 13)
    assert starts[0] == pd.Timestamp("2001-05-31")
    assert starts[-1] == pd.Timestamp("2001-06-30")
    assert weights == pytest.approx(np.array([4, 5, 5, 5, 5, 5, 1]) / 30)
    assert duration == 30
    assert position == 46
    with pytest.raises(ValueError):
        window_layout("2001-05", "2001-04")


def test_pool_retains_boundary_state_not_end_of_overlapping_bin():
    pool = ReplayPool(per_lead=2, max_age=3)
    states = [
        torch.full((1, 2, 1, 1, 1), float(i), requires_grad=True) for i in range(8)
    ]
    pool.add_next("2001-05", "2001-05", states, 10)
    state = pool.draw(np.random.default_rng(1), 12)
    assert state["month"] == "2001-06" and state["bin_index"] == 6
    assert (state["latent"] == 6).all()
    assert not state["latent"].requires_grad
    assert pool.draw(np.random.default_rng(1), 14) is None
    pool.add_next("2013-07", "2013-07", states, 15)
    assert not pool.entries[1]  # No August target or held-out months.


def test_pool_is_bounded_and_serialization_preserves_draws():
    pool = ReplayPool(per_lead=2)
    states = [torch.ones(1, 2, 1, 1, 1) for _ in range(8)]
    for i in range(9):
        pool.add_next("2001-05", "2001-05", states, i)
    assert len(pool.entries[1]) == 2
    payload = io.BytesIO()
    torch.save(pool.state_dict(), payload)
    payload.seek(0)
    restored = ReplayPool(per_lead=2)
    restored.load_state_dict(torch.load(payload, weights_only=False))
    assert (
        restored.draw(np.random.default_rng(4), 9)["created_step"]
        == pool.draw(np.random.default_rng(4), 9)["created_step"]
    )


def test_chain_cannot_extend_its_original_model_age_or_training_horizon():
    pool = ReplayPool(max_age=10)
    states = [torch.ones(1, 2, 1, 1, 1) for _ in range(8)]
    pool.add_next("2001-05", "2001-06", states, 15, origin_step=3)
    assert pool.draw(np.random.default_rng(1), 16) is None
    # A leap-year twelve-month target would exceed the 73-step cap.
    pool.add_next("1999-05", "2000-03", states, 17)
    assert not pool.entries[11]
    # The corresponding non-leap year exactly fits day 365.
    pool.add_next("2001-05", "2002-03", states, 17)
    assert pool.entries[11][0]["month"] == "2002-04"


def test_extension_and_replay_schedules_match_authorized_mix():
    from collections import Counter

    from scripts.run_monthly_replay import phase_settings

    assert Counter(phase_settings(i, "extend", "A")[1] for i in range(2000, 4000)) == {
        "obs": 1000,
        "om4": 1000,
    }
    assert Counter(phase_settings(i, "replay", "A")[1] for i in range(2000)) == {
        "obs": 1000,
        "om4": 1000,
    }
    assert all(phase_settings(i, "extend", "E")[1] == "obs" for i in range(2000, 5000))
    assert Counter(phase_settings(i, "fresh-4k", "A")[1] for i in range(4000)) == {
        "obs": 2000,
        "om4": 2000,
    }
    for mode, start in (("extend", 4000), ("replay", 2000), ("fresh-4k", 4000)):
        assert all(
            phase_settings(i, mode, "A")[1] == "obs" for i in range(start, start + 1000)
        )
        assert phase_settings(start + 999, mode, "A")[2] == pytest.approx(1e-6)
        assert phase_settings(start - 1, mode, "A")[2] == 1e-4


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.init_parameter = torch.nn.Parameter(torch.tensor(0.3))
        self.processor = TinyProcessor()
        self.decoder = TinyDecoder()

    def adapt(self, atmosphere):
        return atmosphere[:, :, :3]

    def initialize(self, fields, available, forcing, context):
        return (
            self.init_parameter.reshape(1, 1, 1, 1, 1).expand(1, 2, 1, 1, 1)
            + fields[:, :19, 38].mean() * 0
        )

    def trajectory(self, fields, available, forcing, contexts):
        return self.processor.rollout(
            self.initialize(fields, available, forcing, contexts),
            forcing[:, 19:],
            contexts[:, 19:],
        )

    @staticmethod
    def monthly_condition(states, weights):
        return sum(s * w for s, w in zip(states, weights, strict=True))


class TinyProcessor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.increment = torch.nn.Parameter(torch.tensor(0.1))

    def rollout(self, initial, forcing, contexts):
        states = [initial]
        for _ in range(forcing.shape[1]):
            states.append(states[-1] + self.increment)
        return states


class TinyDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.records = []

    def loss(self, latent, target, support, area, generator, **kwargs):
        self.records.append((latent.detach().clone(), kwargs["position_days"]))
        return (latent.mean() - target[support].mean()).square()


def fixture():
    sample = dict(
        surface=torch.ones(1, 26, 2, 2, 4),
        validity=torch.ones(1, 26, 2, 2, 4),
        atmosphere=torch.ones(1, 26, 8, 2, 4),
        contexts=torch.ones(1, 26, 5, 2, 4),
        interior=torch.ones(1, 1, 2, 4),
        month_weights=torch.tensor([5 / 31] * 6 + [1 / 31]),
        name="2001-05",
    )
    from pathlib import Path

    data = SimpleNamespace(
        root=Path("/unused"),
        device="cpu",
        mean=torch.zeros(1, 1, 77, 1, 1),
        std=torch.ones(1, 1, 77, 1, 1),
        ts_indices=[39],
        ts_mask=torch.ones(1, 2, 4, dtype=torch.bool),
        area=torch.ones(2, 1),
        load=lambda _: sample,
        tensor=lambda x: torch.as_tensor(x, dtype=torch.float32),
    )
    return data, sample


def test_capture_is_detached_without_changing_original_objective():
    model = TinyModel()
    data, sample = fixture()
    a, _ = observation_loss(
        model, data, sample, "surface", torch.Generator().manual_seed(9)
    )
    captured: list[torch.Tensor] = []
    b, _ = observation_loss(
        model,
        data,
        sample,
        "surface",
        torch.Generator().manual_seed(9),
        capture=lambda states: captured.extend(s.detach().clone() for s in states),
    )
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert len(captured) == 8
    assert all(not s.requires_grad for s in captured)


def test_replay_never_reinitializes_forecast_and_truncates_gradient():
    model = TinyModel()
    data, sample = fixture()
    sample["name"] = "2001-06"
    later = dict(
        surface=torch.full((1, 7, 2, 2, 4), 2.0),
        validity=torch.ones(1, 7, 2, 2, 4),
        atmosphere=torch.ones(1, 7, 8, 2, 4),
        contexts=torch.ones(1, 7, 5, 2, 4),
    )
    intervals = SimpleNamespace(load=lambda *_: later)
    cached_latent = torch.full((1, 2, 1, 1, 1), 4.0, requires_grad=True)
    entry = dict(
        origin="2001-05-01",
        month="2001-06",
        bin_index=6,
        latent=cached_latent,
    )
    loss, _ = replay_loss(
        model, data, intervals, entry, torch.Generator().manual_seed(3)
    )
    before = copy.deepcopy(model.decoder.records)
    assert before[0][1] == 46
    assert before[1][1] in [32.5 + 5 * i for i in range(7)]
    assert before[2][1] == -2.5
    assert before[0][0].mean() > 4
    assert before[2][0].mean() == pytest.approx(0.3)
    loss.backward()
    assert cached_latent.grad is None
    assert model.processor.increment.grad is not None
    assert model.init_parameter.grad is not None
    assert model.processor.increment.grad.abs() > 0
    assert model.init_parameter.grad.abs() > 0
    sample["surface"][:, :19] = (
        100  # Changes reconstruction history, never replay states.
    )
    model.decoder.records = []
    replay_loss(model, data, intervals, entry, torch.Generator().manual_seed(3))
    for index in (0, 1):
        torch.testing.assert_close(
            before[index][0], model.decoder.records[index][0], rtol=0, atol=0
        )
    entry["month"] = "2013-11"
    with pytest.raises(ValueError, match="outside"):
        replay_loss(model, data, intervals, entry, torch.Generator())
