# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from samudra.experiments.observation_model import ObservationTransfer
from samudra.experiments.observation_state_diagnostics import (
    StateIntervention,
    intervene,
)


def test_interventions_preserve_observed_channels_and_source_state():
    source = torch.randn(1, 2, 77, 2, 2)
    original = source.clone()
    velocity = intervene(source, "zero-velocity")
    assert torch.count_nonzero(velocity[:, :, :38]) == 0
    torch.testing.assert_close(velocity[:, :, 38:], source[:, :, 38:])
    deep = intervene(source, "mean-deep-ts")
    retained = list(range(52)) + list(range(57, 71)) + [76]
    torch.testing.assert_close(deep[:, :, retained], source[:, :, retained])
    assert torch.count_nonzero(deep[:, :, 52:57]) == 0
    assert torch.count_nonzero(deep[:, :, 71:76]) == 0
    torch.testing.assert_close(source, original)
    with pytest.raises(ValueError):
        intervene(source, "unknown")


def test_unmodified_diagnostic_matches_native_forecast():
    model: Any = SimpleNamespace(
        initialize=lambda *args: torch.ones(1, 2, 77, 2, 2),
        adapt=lambda a: a[:, :, :3],
        evolution=lambda state, forcing, context, mask, steps: (
            state[:, -1] + forcing[:, 0, :1]
        ),
        call=lambda fn, *args: fn(*args),
    )
    args = (
        torch.zeros(1, 19, 2, 2, 2),
        torch.randn(1, 25, 8, 2, 2),
        torch.zeros(1, 25, 5, 2, 2),
        torch.ones(77, 2, 2),
        torch.ones(1, 19, 2, 2, 2),
    )
    expected = ObservationTransfer.forecast(model, *args)
    actual = StateIntervention(model, "none").forecast(*args)
    for a, b in zip(expected, actual, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
