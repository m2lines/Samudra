# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest
import torch

from samudra.observations.config import ObservationTrainConfig
from samudra.observations.replay import ReplayEntry, ReplayPool, states_at
from samudra.observations.state import TrainingState


def entry(month=1, origin_update=0, source="obs"):
    return ReplayEntry(
        source,
        "2000-01",
        month,
        6 * month,
        origin_update,
        torch.ones(1, 2, 3, 2, 2, requires_grad=True) * month,
    )


def test_exact_source_allocation_and_cooldown():
    from collections import Counter

    cfg = ObservationTrainConfig.model_validate(
        {
            "experiment": {"name": "test", "wandb": {}},
            "data": {"om4": "om4", "cm4": "cm4"},
            "mix": "om4-cm4-obs",
        }
    )
    assert Counter(cfg.source_at(i) for i in range(2000)) == {
        "obs": 1000,
        "om4": 500,
        "cm4": 500,
    }
    assert all(cfg.source_at(i) == "obs" for i in range(2000, 3000))
    assert cfg.learning_rate_at(1999) == 1e-4
    assert cfg.learning_rate_at(2000) < 1e-4
    assert cfg.learning_rate_at(2999) == pytest.approx(1e-6)


def test_overlap_boundary_uses_two_correct_states_and_detaches_storage():
    initial = torch.tensor([-2.0, -1.0]).reshape(1, 2, 1, 1, 1)
    predicted = torch.arange(7.0).reshape(1, 7, 1, 1, 1).requires_grad_()
    # Jan31 requires7 predictions; Feb starts inside bin6. Replay from [4,5].
    states = states_at(initial, predicted, 6)
    torch.testing.assert_close(states.flatten(), torch.tensor([4.0, 5.0]))
    pool = ReplayPool("obs", 0)
    item = entry()
    item.states = states
    pool.add(item)
    saved = pool.draw(0)
    assert saved is not None
    assert saved.states.grad_fn is None and not saved.states.requires_grad
    with torch.no_grad():
        predicted.add_(100)
    torch.testing.assert_close(saved.states.flatten(), torch.tensor([4.0, 5.0]))


def test_source_isolation_age_and_exact_rng_resume():
    pool = ReplayPool("obs", 13, per_month=2, max_age=250)
    with pytest.raises(ValueError, match="mix sources"):
        pool.add(entry(source="cm4"))
    for month in range(1, 12):
        pool.add(entry(month))
    restored = ReplayPool("obs", 99, per_month=2, max_age=250)
    restored.load_state_dict(pool.state_dict())

    def draw_months(candidate):
        months = []
        for _ in range(100):
            selected = candidate.draw(250)
            assert selected is not None
            months.append(selected.month)
        return months

    assert draw_months(pool) == draw_months(restored)
    assert pool.draw(251) is None


def test_optimizer_ema_rng_replay_next_update_exact(tmp_path):
    torch.manual_seed(11)
    model = torch.nn.Linear(2, 2)
    state = TrainingState(model, torch.optim.AdamW(model.parameters()), ("obs", "om4"))
    state.pools["obs"].add(entry())
    manifest = {"experiment": "resume-test"}

    def update(current):
        current.optimizer.zero_grad(set_to_none=True)
        x = torch.rand(2, 2) * current.rng.random()
        loss = current.model(x).square().mean()
        loss.backward()
        current.optimizer.step()
        current.ema(current.model)
        current.updates += 1
        return loss.detach(), current.pools["obs"].draw(current.updates).month

    update(state)
    state.save(tmp_path / "checkpoint.pt", manifest)
    expected = update(state)
    expected_parameters = deepcopy(model.state_dict())
    resumed_model = torch.nn.Linear(2, 2)
    resumed = TrainingState(
        resumed_model, torch.optim.AdamW(resumed_model.parameters()), ("obs", "om4")
    )
    resumed.restore(tmp_path / "checkpoint.pt", manifest)
    actual = update(resumed)
    assert actual == expected
    for name, value in resumed_model.state_dict().items():
        torch.testing.assert_close(value, expected_parameters[name], atol=0, rtol=0)
    assert state.ema == resumed.ema
    with pytest.raises(ValueError, match="manifest differs"):
        resumed.restore(tmp_path / "checkpoint.pt", {"experiment": "wrong"})


def test_ema_context_restores_after_evaluator_error():
    model = torch.nn.Linear(2, 2)
    state = TrainingState(model, torch.optim.Adam(model.parameters()), ("obs",))
    with torch.no_grad():
        model.weight.add_(10)
    before = model.weight.detach().clone()
    with pytest.raises(RuntimeError, match="evaluation failed"):
        with state.averaged_model():
            assert not torch.equal(model.weight, before)
            raise RuntimeError("evaluation failed")
    torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
