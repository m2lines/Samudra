# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import torch

from samudra.experiments.surface_state import (
    Evolution,
    Forecast,
    Initializer,
    balanced_loss,
)
from samudra.experiments.surface_wave import training_batches

NAMES = ["uo_0", "vo_0", "thetao_0", "thetao_1", "so_0", "zos"]


def setup_case():
    torch.set_num_threads(1)
    torch.manual_seed(71)
    history = torch.randn(2, 6 * len(NAMES), 16, 32)
    context = torch.randn(2, 5, 16, 32)
    mask = torch.ones(len(NAMES), 16, 32, dtype=torch.bool)
    mask[:, :2] = False
    forcing = torch.randn(2, 6, 3, 16, 32)
    return history, context, mask, forcing


def test_initializer_cannot_read_interiors_and_preserves_observed_surfaces():
    history, context, mask, _ = setup_case()
    model = Initializer(NAMES, [8, 16]).eval()
    original = model(history, context, mask)
    changed = history.clone().reshape(2, 6, len(NAMES), 16, 32)
    changed[:, :, [0, 1, 3, 4]] += 1000
    actual = model(changed.flatten(1, 2), context, mask)
    torch.testing.assert_close(actual, original)
    expected = history.reshape(2, 6, len(NAMES), 16, 32)[:, -2:, [2, 5]] * mask[[2, 5]]
    torch.testing.assert_close(actual[:, :, [2, 5]], expected)
    assert (actual[..., :2, :] == 0).all()


def test_direct_cannot_see_later_forcing_and_joint_gradient_reaches_initializer():
    history, context, mask, forcing = setup_case()
    model = Forecast(
        Initializer(NAMES, [8, 16]), Evolution(len(NAMES), [8, 16], "direct")
    ).eval()
    result, _ = model(history, forcing, context, mask, "inferred", [2])
    changed = forcing.clone()
    changed[:, 2:] += 1000
    actual, _ = model(history, changed, context, mask, "inferred", [2])
    torch.testing.assert_close(actual, result)
    loss = balanced_loss(result, torch.zeros_like(result), mask.float(), NAMES)
    loss.backward()
    assert (
        sum(
            float(p.grad.abs().sum())
            for p in model.initializer.parameters()
            if p.grad is not None
        )
        > 0
    )


def test_autoregression_does_not_read_initial_hidden_truth_in_inferred_mode():
    history, context, mask, forcing = setup_case()
    model = Forecast(
        Initializer(NAMES, [8, 16]), Evolution(len(NAMES), [8, 16], "ar")
    ).eval()
    original, _ = model(history, forcing, context, mask, "inferred", [1, 3])
    changed = history.clone().reshape(2, 6, len(NAMES), 16, 32)
    changed[:, :, [0, 1, 3, 4]] += 1000
    actual, _ = model(changed.flatten(1, 2), forcing, context, mask, "inferred", [1, 3])
    torch.testing.assert_close(actual, original)
    original.square().mean().backward()
    assert any(p.grad is not None for p in model.initializer.parameters())


def test_loss_excludes_land_and_balances_variable_groups():
    prediction = torch.zeros(1, 1, len(NAMES), 2, 2)
    weights = torch.ones(len(NAMES), 2, 2)
    weights[:, 0] = 0
    prediction[..., 0, :] = 1000
    assert balanced_loss(prediction, torch.zeros_like(prediction), weights, NAMES) == 0
    prediction[:, :, [2, 3], 1] = 1
    torch.testing.assert_close(
        balanced_loss(prediction, torch.zeros_like(prediction), weights, NAMES),
        torch.tensor(0.2),
    )


def test_training_sampler_partitions_without_duplicates_and_resumes():
    ranks = [training_batches(101, 2, 4, r, 1729, 3) for r in range(4)]
    flattened = [i for rank in ranks for batch in rank for i in batch]
    assert len(flattened) == len(set(flattened)) == 96
    assert len({len(rank) for rank in ranks}) == 1
    assert ranks[0][3:] == training_batches(101, 2, 4, 0, 1729, 3)[3:]


def test_initializer_preserves_float32_surfaces_under_autocast():
    history, context, mask, _ = setup_case()
    model = Initializer(NAMES, [8, 16]).eval()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(history, context, mask)
    expected = history.reshape(2, 6, len(NAMES), 16, 32)[:, -2:, [2, 5]] * mask[[2, 5]]
    torch.testing.assert_close(output[:, :, [2, 5]], expected, rtol=0, atol=0)
