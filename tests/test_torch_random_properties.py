# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Hypothesis controls torch's global generator, so torch draws replay with their example."""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st


def torch_draws():
    """Each example's input with the first value it draws from torch."""
    seen = []

    @settings(max_examples=5, derandomize=True, database=None)
    @given(st.integers(0, 3))
    def record(value):
        seen.append((value, torch.rand(()).item()))

    record()
    return seen


def test_torch_draws_depend_only_on_the_hypothesis_seed():
    # Unregistered, the second run continues torch's stream and draws new values.
    assert torch_draws() == torch_draws()
