# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import torch

from samudra.experiments.initializer_diagnostic_wave import selected_loss, step_indices


def test_field_objective_excludes_other_channels_and_dry_cells():
    names = ["thetao_0", "so_9", "uo_9"]
    p = torch.zeros(2, 2, 3, 2, 2, requires_grad=True)
    t = torch.ones_like(p) * 100
    t[:, :, 1] = 2
    t[:, :, 1, 0, 0] = 999
    weights = torch.ones(3, 2, 2)
    weights[:, 0, 0] = 0
    loss = selected_loss(p, t, weights, names, "field", "so_9")
    assert loss.item() == 4
    loss.backward()
    assert p.grad is not None
    assert torch.count_nonzero(p.grad[:, :, 0]) == 0
    assert torch.count_nonzero(p.grad[:, :, 2]) == 0
    assert torch.count_nonzero(p.grad[:, :, :, 0, 0]) == 0
    assert torch.count_nonzero(p.grad[:, :, 1]) == 12


def test_matched_sampler_resumes_and_tiny_sets_stay_in_scope():
    population = [
        3,
        80,
        150,
        400,
        600,
        800,
        900,
        1100,
        1300,
        1500,
        1700,
        1900,
        2100,
        2300,
        2500,
        2700,
    ]
    uninterrupted = [step_indices(population, 1729, i, 8) for i in range(10)]
    assert uninterrupted[5:] == [
        step_indices(population, 1729, i, 8) for i in range(5, 10)
    ]
    assert all(
        len(set(batch)) == 8 and set(batch) <= set(population)
        for batch in uninterrupted
    )
    assert step_indices([13], 1729, 5, 8) == [13] * 8
