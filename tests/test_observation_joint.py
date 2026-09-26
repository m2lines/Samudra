# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import copy
import time
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from samudra.experiments import observation_joint as joint
from samudra.experiments.task_schedule import TaskSchedule


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.initializer = nn.Linear(1, 1, bias=False)
        self.evolution = nn.Linear(1, 1, bias=False)
        self.adapter = nn.Linear(1, 1, bias=False)

    def loss(self, observation):
        state = self.evolution(self.initializer(torch.ones(1, 1)))
        if observation:
            state = state + self.adapter(torch.ones(1, 1))
        return (state - 2).square().mean()


def toy_pilot(monkeypatch):
    monkeypatch.setattr(torch, "autocast", lambda *a, **k: nullcontext())
    monkeypatch.setattr(joint, "om4_objective", lambda model, *a: model.loss(False))
    torch.manual_seed(42)
    pilot: Any = joint.JointPilot.__new__(joint.JointPilot)
    pilot.model = Toy()
    pilot.schedule = TaskSchedule(2, 2, "sequential")
    pilot.args = SimpleNamespace(
        seed=1729,
        accumulate=2,
        core_lr=0.01,
        om4_lr=0.01,
        warmup_steps=0,
        reconstruction_weight=0.1,
    )
    pilot.training = list(range(5))
    pilot.om4 = SimpleNamespace(trainset=list(range(7)))
    pilot.data = SimpleNamespace(load=lambda i: i)
    pilot.objective = lambda *a: pilot.model.loss(True)
    pilot.optimizer = torch.optim.AdamW(pilot.model.parameters(), lr=0.01)
    pilot.completed = 0
    pilot.elapsed_seconds = 0.0
    pilot.started = time.monotonic()
    pilot.emit = lambda record: None
    return pilot


def test_single_optimizer_and_task_gradient_routing(monkeypatch):
    pilot = toy_pilot(monkeypatch)
    optimizer = pilot.optimizer
    adapter_before = pilot.model.adapter.weight.detach().clone()
    pilot.train_update()
    pilot.train_update()
    torch.testing.assert_close(
        pilot.model.adapter.weight, adapter_before, rtol=0, atol=0
    )
    assert pilot.model.adapter.weight not in optimizer.state
    assert optimizer.state[pilot.model.initializer.weight]["step"] == 2
    pilot.train_update()
    assert pilot.optimizer is optimizer
    assert optimizer.state[pilot.model.initializer.weight]["step"] == 3
    assert optimizer.state[pilot.model.adapter.weight]["step"] == 1
    assert not torch.equal(pilot.model.adapter.weight, adapter_before)


def test_resumed_task_boundary_matches_uninterrupted(monkeypatch):
    pilot = toy_pilot(monkeypatch)
    pilot.train_update()
    pilot.train_update()
    saved = copy.deepcopy(
        (pilot.model.state_dict(), pilot.optimizer.state_dict(), pilot.completed)
    )
    pilot.train_update()
    expected = copy.deepcopy(pilot.model.state_dict())
    fresh = toy_pilot(monkeypatch)
    fresh.model.load_state_dict(saved[0])
    fresh.optimizer.load_state_dict(saved[1])
    fresh.completed = saved[2]
    fresh.train_update()
    for name, value in expected.items():
        torch.testing.assert_close(
            fresh.model.state_dict()[name], value, rtol=0, atol=0
        )


def test_nonfinite_does_not_advance_optimizer(monkeypatch):
    pilot = toy_pilot(monkeypatch)
    monkeypatch.setattr(
        joint, "om4_objective", lambda model, *a: model.loss(False) * float("nan")
    )
    with pytest.raises(FloatingPointError):
        pilot.train_update()
    assert pilot.completed == 0
    assert not pilot.optimizer.state
