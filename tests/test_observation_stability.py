# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Protect diagnostic isolation and checkpoint migration contracts."""

import copy
from typing import Any

import numpy as np
import pytest
import torch

from samudra.experiments.task_schedule import TaskSchedule
from scripts.run_observation_stability import check_migration, preserve_execution


def test_diagnostics_restore_rng_and_individual_module_modes(monkeypatch):
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout())
    model.train()
    model[1].eval()
    cuda_state = torch.tensor([1], dtype=torch.uint8)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: cuda_state.clone())
    monkeypatch.setattr(
        torch.cuda, "set_rng_state", lambda x: torch.testing.assert_close(x, cuda_state)
    )
    torch.manual_seed(12)
    np.random.seed(12)
    cpu = torch.get_rng_state().clone()
    numpy = copy.deepcopy(np.random.get_state())
    assert isinstance(numpy, tuple)
    with pytest.raises(RuntimeError), preserve_execution(model):
        model.eval()
        torch.rand(10)
        np.random.rand(10)
        raise RuntimeError("diagnostic failure")
    torch.testing.assert_close(torch.get_rng_state(), cpu)
    restored = np.random.get_state()
    assert isinstance(restored, tuple)
    np.testing.assert_array_equal(restored[1], numpy[1])
    assert model.training and model[0].training and not model[1].training


def test_migration_allows_reporting_changes_but_rejects_objective_or_batch_changes():
    old = dict(
        core_lr=1e-4,
        accumulate=8,
        reconstruction_weight=0.1,
        selection_metric="integrated-spectral",
        output="old",
        wandb_mode="offline",
    )
    new = dict(
        old,
        selection_metric="rmse-only",
        output="new",
        wandb_mode="online",
        recent_only=True,
        iap_grid=None,
    )
    check_migration(old, new)
    for key, value in (
        ("accumulate", 16),
        ("core_lr", 3e-5),
        ("reconstruction_weight", 0.2),
    ):
        with pytest.raises(ValueError, match=key):
            check_migration(old, dict(new, **{key: value}))


def test_early_diagnostic_branch_and_fork_counts():
    schedule = TaskSchedule(2000, 2000, "mixed")
    assert schedule.counts(1) == dict(om4=1, observation=0)
    assert schedule.counts(2667) == dict(om4=1667, observation=1000)
    assert schedule.counts(3000) == dict(om4=1782, observation=1218)


def test_observation_component_logging_preserves_loss_and_gradients():
    from types import SimpleNamespace

    from samudra.experiments.observation_pilot import Pilot

    weight = torch.nn.Parameter(torch.tensor(2.0))
    pilot: Any = Pilot.__new__(Pilot)
    pilot.args = SimpleNamespace(completion_weight=0.0, reconstruction_weight=0.1)
    pilot.arguments = lambda sample: ()
    pilot.model = SimpleNamespace(
        forecast=lambda: (weight.square(), None),
        reconstruct_month=lambda weights: weight.pow(3),
    )
    pilot.data = SimpleNamespace(
        forecast_loss=lambda prediction, sample: prediction,
        interior_loss=lambda reconstructed, sample: reconstructed,
    )
    sample = {"month_weights": None}
    plain = pilot.objective(sample, "joint")
    plain.backward()
    assert weight.grad is not None
    expected_gradient = weight.grad.clone()
    weight.grad = None
    records: list[dict[str, torch.Tensor]] = []
    pilot.loss_observer = records.append
    observed = pilot.objective(sample, "joint")
    observed.backward()
    torch.testing.assert_close(observed, plain, rtol=0, atol=0)
    torch.testing.assert_close(weight.grad, expected_gradient, rtol=0, atol=0)
    torch.testing.assert_close(sum(records[0].values()), observed.detach())
    assert all(not value.requires_grad for value in records[0].values())
