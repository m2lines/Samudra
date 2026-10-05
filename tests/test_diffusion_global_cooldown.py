# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest
import torch

from samudra.experiments.diffusion_global_cooldown import (
    CooldownSchedule,
    validate_parent,
)
from samudra.experiments.task_schedule import sample_indices


def test_cooldown_preserves_original_exposure_and_continues_samples():
    schedule = CooldownSchedule()
    assert schedule.counts(6949) == dict(om4=4949, observation=2000)
    assert schedule.counts(16000) == dict(om4=8000, observation=8000)
    assert schedule.counts(18000) == dict(om4=8250, observation=9750)
    assert schedule.task(16000) == "om4"
    assert all(schedule.task(i) == "observation" for i in range(16001, 16008))
    counters = dict(om4=8000, observation=8000)
    for index in range(16000, schedule.total):
        task = schedule.task(index)
        assert schedule.counts(index) == counters
        counters[task] += 1
    assert counters == schedule.counts(schedule.total)
    rates = [schedule.learning_rate(i) for i in range(16000, schedule.total)]
    assert rates[0] == pytest.approx(1e-4)
    assert rates[-1] == pytest.approx(1e-6)
    assert all(a > b for a, b in zip(rates, rates[1:]))
    with pytest.raises(ValueError, match="outside"):
        schedule.learning_rate(18000)


def test_resume_preserves_optimizer_rng_lr_and_example_sequence():
    # Exercise interruption in the middle of a cooldown; a fresh optimizer or
    # an invocation-relative LR/sample counter must not reproduce this result.
    schedule = CooldownSchedule(updates=16)
    torch.manual_seed(43)
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def update(net, opt, index):
        task = schedule.task(index)
        count = schedule.counts(index)[task]
        ids = sample_indices(243, 1729, count, 8)
        x = torch.randn(8, 3) + torch.tensor(ids).float()[:, None] / 243
        opt.param_groups[0]["lr"] = schedule.learning_rate(index)
        opt.zero_grad(set_to_none=True)
        net(x).square().mean().backward()
        opt.step()

    for i in range(16000, 16005):
        update(model, optimizer, i)
    state = copy.deepcopy(
        dict(model=model.state_dict(), optimizer=optimizer.state_dict())
    )
    rng = torch.get_rng_state()
    for i in range(16005, schedule.total):
        update(model, optimizer, i)
    resumed = torch.nn.Linear(3, 1)
    resumed.load_state_dict(state["model"])
    other = torch.optim.AdamW(resumed.parameters(), lr=1e-4)
    other.load_state_dict(state["optimizer"])
    torch.set_rng_state(rng)
    for i in range(16005, schedule.total):
        update(resumed, other, i)
    for expected, actual in zip(model.parameters(), resumed.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for key, values in optimizer.state_dict()["state"].items():
        for name, expected in values.items():
            torch.testing.assert_close(
                other.state_dict()["state"][key][name], expected, rtol=0, atol=0
            )


def test_cooldown_rejects_incomplete_or_weights_only_parent():
    contract = dict(producer="original", effective_batch=8, learning_rate=1e-4)
    saved = dict(
        contract=contract,
        step=16000,
        counts=dict(om4=8000, observation=8000),
        optimizer=dict(param_groups=[dict(lr=1e-4)]),
        cpu_rng="saved",
        cuda_rng="saved",
    )
    validate_parent(saved, contract)
    with pytest.raises(ValueError, match="endpoint"):
        validate_parent(dict(saved, step=15999), contract)
    with pytest.raises(ValueError, match="contract"):
        validate_parent(saved, dict(contract, effective_batch=4))
    with pytest.raises(ValueError, match="optimizer and RNG"):
        validate_parent({k: v for k, v in saved.items() if k != "optimizer"}, contract)
    with pytest.raises(ValueError, match="optimizer LR"):
        validate_parent(
            dict(saved, optimizer=dict(param_groups=[dict(lr=1e-5)])), contract
        )
