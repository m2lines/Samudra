# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Scientific invariants of the matched member-state pilot trainer."""

from types import SimpleNamespace

import pytest
import torch

from samudra.experiments.stochastic_state_train import (
    coherent_reconstruction,
    counts,
    learning_rate,
)


def test_alternating_counts_and_cooldown():
    assert counts(0) == {"om4": 0, "observation": 0}
    assert counts(3) == {"om4": 2, "observation": 1}
    assert counts(2000) == {"om4": 1000, "observation": 1000}
    assert learning_rate(1599, 2000) == 1e-4
    assert learning_rate(1600, 2000) == 1e-4
    assert learning_rate(1999, 2000) == pytest.approx(1e-6)
    rates = [learning_rate(i, 2000) for i in range(1600, 2000)]
    assert rates == sorted(rates, reverse=True)


def test_monthly_reconstruction_carries_one_initial_member_draw():
    class Toy:
        history = 19

        def adapt(self, atmosphere):
            return atmosphere

        def encode_native(self, surface, past, context, mask, validity, task):
            assert surface[0, 0, 0, 0, 0] == 1
            assert surface[0, -1, 0, 0, 0] == 19
            return torch.ones(1, 2, 1, 1, 1), None, None

        def rollout_members(self, initial, forcing, contexts, generator, members, task):
            assert forcing.shape[1] == 2
            state = initial[None].repeat(2, 1, 1, 1, 1, 1)
            state[1] *= 3
            return [state, state + 1, state + 2]

        def decode_members(self, state, mask, known=None, anchor=None):
            return state

    data = SimpleNamespace(mask=torch.ones(1, 1, 1))
    sample = {
        "surface": torch.arange(22).reshape(1, 22, 1, 1, 1).float(),
        "atmosphere": torch.zeros(1, 22, 3, 1, 1),
        "contexts": torch.zeros(1, 22, 5, 1, 1),
        "validity": torch.ones(1, 22, 1, 1, 1),
        "month_weights": torch.tensor([0.2, 0.3, 0.5]),
    }
    monthly, bound = coherent_reconstruction(Toy(), data, sample, torch.Generator())
    # Each member retains its own +2 difference throughout the entire month.
    torch.testing.assert_close(monthly.flatten(), torch.tensor([2.3, 4.3]))
    assert torch.isfinite(bound)
