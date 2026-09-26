# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from samudra.experiments.observation_mechanisms import forecast, replace_hidden


def test_seasonal_replacement_preserves_both_surface_histories():
    original = torch.arange(154.0).reshape(1, 2, 77, 1, 1)
    replacement = torch.full_like(original, -1.0)
    changed = replace_hidden(original, replacement)
    torch.testing.assert_close(changed[:, :, [38, 76]], original[:, :, [38, 76]])
    assert (changed[:, :, :38] == -1).all()
    velocity = replace_hidden(original, replacement, True)
    torch.testing.assert_close(velocity[:, :, 38:], original[:, :, 38:])
    torch.testing.assert_close(original.flatten(), torch.arange(154.0))
    with pytest.raises(ValueError):
        replace_hidden(original, replacement[:, :1])


class Toy:
    def initialize(self, surface, atmosphere, context, mask, validity):
        assert surface.shape[1] == atmosphere.shape[1] == 19
        return torch.ones(1, 2, 77, 1, 1)

    def adapt(self, atmosphere):
        return atmosphere

    def call(self, function, *args):
        return function(*args)

    def evolution(self, states, forcing, context, mask, lead):
        # Surface reads a hidden channel, making intervention timing observable.
        result = states[:, -1] + forcing[:, 0, :1]
        result[:, 38] += states[:, -1, 0]
        return result


def test_intervention_timing_and_noop():
    arguments = (
        torch.zeros(1, 19, 2, 1, 1),
        torch.ones(1, 27, 8, 1, 1),
        torch.zeros(1, 27, 5, 1, 1),
        torch.ones(77, 1, 1),
        torch.ones(1, 19, 2, 1, 1),
    )
    means = {
        "state_mean": np.zeros((12, 2, 77, 1, 1), np.float32),
        "forcing_mean": np.zeros((12, 8, 1, 1), np.float32),
    }
    months = np.zeros(27, int)
    baseline = forecast(Toy(), arguments, months, means, "baseline")
    late = forecast(Toy(), arguments, months, means, "day30-hidden-seasonal")
    torch.testing.assert_close(late[:, :6], baseline[:, :6], rtol=0, atol=0)
    assert not torch.equal(late[:, 6], baseline[:, 6])
    initial = forecast(Toy(), arguments, months, means, "initial-hidden-seasonal")
    assert not torch.equal(initial[:, 0], baseline[:, 0])
    forcing = forecast(Toy(), arguments, months, means, "seasonal-future-forcing")
    assert not torch.equal(forcing[:, 0], baseline[:, 0])
    assert (arguments[1] == 1).all()
    with pytest.raises(ValueError):
        forecast(Toy(), arguments, months, means, "unknown")
