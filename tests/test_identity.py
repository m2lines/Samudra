# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from samudra.datasets import TrainData
from samudra.identity import set_identity_target
from samudra.utils.ctx import GridContext


def make_train_data(prognostic_channels=3, label_channels=3, steps=1):
    context = GridContext(
        label_mask=torch.ones(label_channels, 4, 8, dtype=torch.bool),
        input_resolution_cpu=(torch.arange(4), torch.arange(8)),
        output_resolution_cpu=(torch.arange(4), torch.arange(8)),
    )
    data = TrainData(prognostic_channels, 2, context)
    for _ in range(steps):
        data.append(
            torch.randn(2, prognostic_channels, 4, 8),
            torch.randn(2, 2, 4, 8),
            torch.randn(2, label_channels, 4, 8),
        )
    return data


def test_set_identity_target_reuses_prognostic_input():
    data = make_train_data()
    prognostic, boundary, _ = data[0]

    target = set_identity_target(data)

    new_prognostic, new_boundary, new_target = data[0]
    assert target is prognostic
    assert new_prognostic is prognostic
    assert new_boundary is boundary
    assert new_target is prognostic


def test_set_identity_target_rejects_multiple_steps():
    with pytest.raises(ValueError, match="exactly one model step"):
        set_identity_target(make_train_data(steps=2))


def test_set_identity_target_rejects_shape_change():
    with pytest.raises(ValueError, match="matching prognostic and output shapes"):
        set_identity_target(make_train_data(prognostic_channels=3, label_channels=2))
