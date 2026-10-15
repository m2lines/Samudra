# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Cooldown must preserve Adam moments while resetting only the weight EMA."""

import copy

import pytest
import torch

from samudra.experiments.diffusion_scratch_obs_cooldown import (
    learning_rate,
    reset_ema,
    update_ema,
    validate_parent,
)


def test_linear_schedule_and_resume_positions():
    assert learning_rate(0, 1000) == pytest.approx(1e-4)
    assert learning_rate(999, 1000) == pytest.approx(1e-6)
    assert learning_rate(499, 1000) - learning_rate(500, 1000) == pytest.approx(
        99e-6 / 999
    )
    with pytest.raises(ValueError):
        learning_rate(1000, 1000)


def test_ema_reset_adam_continuation_and_resume():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.tensor([[0.4, -0.8]])
    model(x).square().sum().backward()
    optimizer.step()
    parent_weights = copy.deepcopy(model.state_dict())
    parent_adam = copy.deepcopy(optimizer.state_dict())
    fork = torch.nn.Linear(2, 1)
    fork.load_state_dict(parent_weights)
    fork_optimizer = torch.optim.AdamW(fork.parameters())
    fork_optimizer.load_state_dict(parent_adam)
    ema = reset_ema(fork)
    for key, value in parent_weights.items():
        torch.testing.assert_close(ema[key], value)
        assert ema[key].data_ptr() != fork.state_dict()[key].data_ptr()
    for m, opt in ((model, optimizer), (fork, fork_optimizer)):
        opt.zero_grad(set_to_none=True)
        m(x).square().sum().backward()
        opt.step()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(fork.state_dict()[key], value)
    update_ema(ema, fork, 0.9)
    for key, value in ema.items():
        torch.testing.assert_close(
            value, 0.9 * parent_weights[key] + 0.1 * fork.state_dict()[key]
        )
    # Resuming retains the accumulated EMA, rather than resetting to current weights.
    resumed = {key: value.clone() for key, value in ema.items()}
    update_ema(ema, fork, 0.9)
    update_ema(resumed, fork, 0.9)
    for key in ema:
        torch.testing.assert_close(ema[key], resumed[key])
    assert all(state["step"].item() == 2 for state in fork_optimizer.state.values())


def test_ema_all_buffers_are_copied():
    model = torch.nn.BatchNorm1d(2)
    ema = reset_ema(model)
    model(torch.randn(3, 2))
    update_ema(ema, model, 0.99)
    assert ema["num_batches_tracked"].item() == 1
    torch.testing.assert_close(ema["running_mean"], model.running_mean)
    torch.testing.assert_close(ema["running_var"], model.running_var)


def test_parent_requires_exact_exposure_and_optimizer():
    parent = dict(
        processor_prediction="residual",
        step=11721,
        counts=dict(om4=7721, observation=4000),
        optimizer={},
        rank_states=[dict(rank=i) for i in range(8)],
    )
    validate_parent(parent, 8)
    with pytest.raises(ValueError):
        validate_parent(parent, 4)
    with pytest.raises(ValueError):
        validate_parent({**parent, "counts": dict(om4=7722, observation=4000)}, 8)
    missing = dict(parent)
    del missing["optimizer"]
    with pytest.raises(ValueError):
        validate_parent(missing, 8)
