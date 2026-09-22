# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from samudra.experiments.surface_adaptation import (
    Adaptation,
    AdaptiveForecast,
    state_fingerprint,
    thermohaline_mse,
)
from samudra.experiments.surface_state import Evolution, Initializer, balanced_loss

NAMES = ["uo_0", "vo_0", "thetao_0", "thetao_1", "so_0", "zos"]


@pytest.mark.parametrize("arm", ["initializer", "evolution", "joint"])
def test_adaptation_changes_only_trainable_component_including_bn_buffers(arm):
    torch.set_num_threads(1)
    torch.manual_seed(71)
    model = AdaptiveForecast(
        Initializer(NAMES, [8, 16]), Evolution(len(NAMES), [8, 16], "ar"), arm
    )
    before = {
        key: state_fingerprint(getattr(model, key))
        for key in ("initializer", "evolution")
    }
    # Validation must not undo the frozen-component training-mode policy.
    model.eval()
    model.train()
    history = torch.randn(2, 6 * len(NAMES), 16, 32)
    context = torch.randn(2, 5, 16, 32)
    mask = torch.ones(len(NAMES), 16, 32, dtype=torch.bool)
    forcing = torch.randn(2, 6, 3, 16, 32)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-5
    )
    prediction, _ = model(
        history, forcing, context, mask, "inferred", [1, 2, 3, 4, 5, 6]
    )
    loss = balanced_loss(prediction, torch.zeros_like(prediction), mask.float(), NAMES)
    loss.backward()
    if arm == "initializer":
        # This gradient must pass through all six applications of frozen evolution.
        assert (
            sum(
                float(p.grad.abs().sum())
                for p in model.initializer.parameters()
                if p.grad is not None
            )
            > 0
        )
        assert all(p.grad is None for p in model.evolution.parameters())
    if arm == "evolution":
        assert all(p.grad is None for p in model.initializer.parameters())
        assert any(p.grad is not None for p in model.evolution.parameters())
    optimizer.step()
    for key in ("initializer", "evolution"):
        module = getattr(model, key)
        frozen = (arm == "initializer" and key == "evolution") or (
            arm == "evolution" and key == "initializer"
        )
        assert (state_fingerprint(module) == before[key]) == frozen
        for layer in module.modules():
            if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
                assert layer.training != frozen


def test_selection_ignores_surface_and_velocity_and_balances_ts_levels():
    names = ["thetao_0", "thetao_1", "thetao_2", "so_0", "uo_0", "vo_0", "zos"]
    errors = torch.tensor([[[1000.0, 2.0, 4.0, 5.0, 1000.0, 1000.0, 1000.0]]])
    torch.testing.assert_close(thermohaline_mse(errors, names), torch.tensor([[4.0]]))


def test_data_order_seed_does_not_change_loaded_initial_model():
    torch.manual_seed(1729)
    first = AdaptiveForecast(
        Initializer(NAMES, [8, 16]), Evolution(len(NAMES), [8, 16], "ar"), "joint"
    )
    checkpoint = copy.deepcopy(first.state_dict())
    torch.manual_seed(1730)
    repeat = AdaptiveForecast(
        Initializer(NAMES, [8, 16]), Evolution(len(NAMES), [8, 16], "ar"), "joint"
    )
    assert state_fingerprint(first) != state_fingerprint(repeat)
    repeat.load_state_dict(checkpoint)
    assert state_fingerprint(first) == state_fingerprint(repeat)


def test_origin_metrics_preserve_paired_identity_and_physical_mse():
    experiment = object.__new__(Adaptation)
    experiment.names = NAMES
    experiment.std = torch.tensor([1.0, 1.0, 2.0, 3.0, 4.0, 1.0])
    dataset = SimpleNamespace(
        sources=[SimpleNamespace(time=SimpleNamespace(values=np.arange(20)))]
    )
    errors = torch.arange(24, dtype=torch.float32).reshape(2, 2, 6)
    records = Adaptation.origin_records(
        experiment,
        errors,
        errors + 1,
        errors + 2,
        [0, 6],
        dataset,
        "inferred",
        "global",
        1,
    )
    assert len(records) == 2 * 2 * 6
    rows = {(r["origin_index"], r["lead_days"], r["variable"]): r for r in records}
    assert len(rows) == len(records)
    assert rows[0, 5, "thetao"]["normalized_mse"] == 3
    assert rows[0, 5, "thetao"]["physical_mse"] == 27
    assert rows[6, 10, "so"]["physical_mse"] == 22 * 16
    assert rows[6, 10, "so"]["origin_time"] == "11"
