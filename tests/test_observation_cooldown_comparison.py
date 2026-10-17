# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import time
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from samudra.experiments.task_schedule import TaskSchedule, sample_indices
from scripts.run_observation_cooldown_comparison import CooldownPilot, CooldownSchedule


def test_tail_exposure_preserves_prefix_and_observation_stream():
    prefix = TaskSchedule(2000, 2000, "mixed")
    for branch in "ABCD":
        schedule = CooldownSchedule(branch)
        for step in range(4001):
            assert schedule.counts(step) == prefix.counts(step)
        tasks = [schedule.task(i) for i in range(4000, 5000)]
        assert tasks.count("om4") == (125 if branch == "A" else 0)
        assert schedule.counts(5000) == (
            dict(om4=2125, observation=2875)
            if branch == "A"
            else dict(om4=2000, observation=3000)
        )
        with pytest.raises(ValueError):
            schedule.task(5000)
    a, b = CooldownSchedule("A"), CooldownSchedule("B")
    ids_a = [
        sample_indices(500, 1729, a.counts(i)["observation"], 8)
        for i in range(4000, 5000)
        if a.task(i) == "observation"
    ]
    ids_b = [
        sample_indices(500, 1729, b.counts(i)["observation"], 8)
        for i in range(4000, 5000)
    ]
    assert ids_a == ids_b[:875]


@pytest.mark.parametrize("branch", list("ABCD"))
def test_lr_and_ema_source_and_restart_contract(tmp_path, monkeypatch, branch):
    monkeypatch.setattr(torch.cuda, "get_rng_state", torch.get_rng_state)
    pilot: Any = CooldownPilot.__new__(CooldownPilot)
    pilot.branch = pilot.policy = branch
    pilot.model = torch.nn.Linear(2, 1)
    pilot.optimizer = torch.optim.AdamW(pilot.model.parameters(), lr=1e-4)
    pilot.completed = 4000
    pilot.schedule = CooldownSchedule(branch)
    pilot.manifest = dict(code_commit="test")
    pilot.elapsed_seconds = 0
    pilot.started = time.monotonic()
    pilot.in_update = False
    pilot.ema = None
    pilot.out = tmp_path
    pilot.resume = tmp_path / "joint-last.pt"
    pilot.checkpoint(pilot.resume)
    saved = torch.load(pilot.resume, weights_only=False)
    retained = {k: v + 2 for k, v in saved["model"].items()}
    source = tmp_path / "source.pt"
    torch.save(
        dict(saved, screen=dict(policy="S0"), ema=retained, ema_updates=1000), source
    )
    pilot.activate_source(source)
    ema: Any = getattr(pilot, "ema")
    for key, value in ema.state.items():
        torch.testing.assert_close(
            value,
            retained[key] if branch == "D" else saved["model"][key],
            rtol=0,
            atol=0,
        )
    assert ema.updates == (1000 if branch == "D" else 0)
    assert pilot.task_learning_rate("observation") == pytest.approx(0.000099901)
    pilot.completed = 4999
    assert pilot.task_learning_rate("observation") == pytest.approx(1e-6)
    pilot.completed = 4000
    pilot.activate_source(source)  # Exact accepted resume preserves branch EMA.
    original = pilot.branch
    pilot.branch = "D" if original != "D" else "B"
    with pytest.raises(ValueError, match="contract"):
        pilot.activate_source(source)


def test_forecast_only_preserves_corrupted_inputs_and_removes_auxiliary_calls(
    monkeypatch,
):
    import scripts.run_observation_cooldown_comparison as module

    pilot: Any = CooldownPilot.__new__(CooldownPilot)
    pilot.branch = "C"
    pilot.args = SimpleNamespace(seed=1729)
    called = []

    def corrupt(sample, seed):
        called.append(seed)
        return dict(sample, masked=True)

    monkeypatch.setattr(module, "corrupt_sample", corrupt)
    pilot.arguments = lambda sample: (sample["masked"],)
    weight = torch.nn.Parameter(torch.tensor(2.0))
    pilot.model = SimpleNamespace(
        forecast=lambda masked: (weight.square() if masked else -weight, None)
    )
    pilot.data = SimpleNamespace(forecast_loss=lambda prediction, sample: prediction)
    loss = pilot.objective(dict(mask_seed=123), "joint")
    loss.backward()
    assert weight.grad is not None
    assert called == [123] and loss.item() == 4 and weight.grad.item() == 4
