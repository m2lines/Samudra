# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Resume, exposure and EMA invariants for the physical-state pilot."""

import copy
import random
from collections import Counter

import numpy as np
import pytest
import torch

from samudra.experiments.physical_state_train import (
    checkpoint_payload,
    configuration,
    counts,
    parser,
    phase_settings,
    restore_checkpoint,
    total_updates,
    update_ema,
    validate_args,
)


def config():
    return configuration(parser().parse_args(["--describe"]))


def test_description_shared_direct_forcing_and_no_replay_claim():
    args = parser().parse_args(["--describe"])
    validate_args(args)
    contract = configuration(args)
    assert contract["forcing_convention"] == "shared-era5"
    assert contract["forcing_channels"] == 8
    assert not contract["forcing_adapter"]
    assert not contract["replay"]
    assert not contract["latent_bound"]
    assert contract["effective_batch"] == 8
    assert contract["training_members"] == 2
    with pytest.raises(SystemExit):
        parser().parse_args(["--forcing-convention", "native-om4"])
    args.describe = False
    with pytest.raises(ValueError, match="requires"):
        validate_args(args)


def test_warmup_mixed_and_observation_only_cooldown_exact_exposures():
    contract = config()
    schedule = [phase_settings(i, contract) for i in range(total_updates(contract))]
    assert len(schedule) == 1200
    assert Counter(task for _, task, _ in schedule) == dict(om4=600, observation=600)
    assert counts(99, contract) == dict(om4=99, observation=0)
    assert counts(101, contract) == dict(om4=101, observation=0)
    assert counts(1100, contract) == dict(om4=600, observation=500)
    assert counts(1200, contract) == dict(om4=600, observation=600)
    assert all(
        phase == "om4_warmup" and task == "om4" for phase, task, _ in schedule[:100]
    )
    assert all(
        phase == "observation_cooldown" and task == "observation"
        for phase, task, _ in schedule[1100:]
    )
    assert all(lr == 1e-4 for _, _, lr in schedule[:1100])
    assert schedule[-1][2] == pytest.approx(1e-6)
    rates = [rate for _, _, rate in schedule[1100:]]
    assert rates == sorted(rates, reverse=True)
    contract.update(om4_warmup_updates=0, obs_cooldown_updates=0)
    assert phase_settings(0, contract)[1] == "om4"
    assert phase_settings(1, contract)[1] == "observation"
    assert counts(1000, contract) == dict(om4=500, observation=500)


def test_resume_restores_adam_ema_task_counts_and_all_cpu_rngs():
    model = torch.nn.Linear(2, 1)
    ema = copy.deepcopy(model).requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    update_ema(ema, model, 100)
    contract = config()
    saved = copy.deepcopy(
        checkpoint_payload(model, optimizer, contract, 3, "run-id", ema)
    )
    expected = (torch.rand(2), np.random.rand(2), random.random())
    with torch.no_grad():
        model.weight.fill_(100)
        ema.weight.fill_(200)
    optimizer.param_groups[0]["lr"] = 1
    assert restore_checkpoint(saved, model, optimizer, contract, ema) == (3, "run-id")
    torch.testing.assert_close(torch.rand(2), expected[0])
    np.testing.assert_equal(np.random.rand(2), expected[1])
    assert random.random() == expected[2]
    torch.testing.assert_close(model.weight, saved["model"]["weight"])
    torch.testing.assert_close(ema.weight, saved["ema"]["weight"])
    assert optimizer.param_groups[0]["lr"] == 1e-4
    assert len(optimizer.state) == 2
    saved["counts"]["observation"] += 1
    with pytest.raises(ValueError, match="exposure"):
        restore_checkpoint(saved, model, optimizer, contract, ema)


def test_ema_half_life_and_buffer_copy():
    model = torch.nn.BatchNorm1d(2)
    ema = copy.deepcopy(model).requires_grad_(False)
    assert model.running_mean is not None
    with torch.no_grad():
        ema.weight.zero_()
        model.weight.fill_(1)
        model.running_mean.fill_(3)
    for _ in range(100):
        update_ema(ema, model, 100)
    torch.testing.assert_close(ema.weight, torch.full((2,), 0.5))
    torch.testing.assert_close(ema.running_mean, model.running_mean)
