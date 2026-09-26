# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn

from samudra.experiments.diffusion_evaluation import EnsembleMeanForecast


class NonlinearForecast(nn.Module):
    stochastic = True

    def forecast(
        self, surface, atmosphere, contexts, mask, validity, *, generator, members
    ):
        initial = torch.randn(members, 1, 2, 4, 8, 12, generator=generator)
        # Nonlinearity makes averaging before dynamics observably wrong.
        trajectories = initial.square()
        return trajectories, initial


def test_point_interface_averages_after_member_dynamics_with_fixed_evaluation_draws():
    model = NonlinearForecast()
    wrapper = EnsembleMeanForecast(model, members=8, seed=71)
    placeholder = torch.zeros(1)
    args = (placeholder,) * 5
    prediction, initial = wrapper.forecast(*args)
    trajectories, initials = model.forecast(
        *args, generator=torch.Generator().manual_seed(71), members=8
    )
    torch.testing.assert_close(prediction, trajectories.mean(0), rtol=0, atol=0)
    torch.testing.assert_close(initial, initials.mean(0), rtol=0, atol=0)
    assert not torch.allclose(prediction, initial.square())
    torch.manual_seed(999)
    repeated, _ = wrapper.forecast(*args)
    torch.testing.assert_close(repeated, prediction, rtol=0, atol=0)
