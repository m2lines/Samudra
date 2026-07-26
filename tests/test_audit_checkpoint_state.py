# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

import pytest
import torch

from scripts.audit_checkpoint_state import (
    _inverse_preservation,
    _metadata,
    _model_state,
    _parameter_summary,
)


def test_checkpoint_state_audit_helpers() -> None:
    checkpoint = {
        "epoch": 3,
        "num_optimizer_updates": 12,
        "model": OrderedDict(
            {
                "module.encoder.weight": torch.tensor([1.0, 2.0]),
                "module.decoder.bias": torch.tensor([3.0]),
                "module.processor.weight": torch.tensor([4.0]),
            }
        ),
    }
    current = _model_state(checkpoint)
    reference = OrderedDict(
        {
            "encoder.weight": torch.tensor([1.0, 2.0]),
            "decoder.bias": torch.tensor([3.0]),
        }
    )

    preservation = _inverse_preservation(current, reference)
    assert preservation["exact"]
    assert preservation["shared_tensors"] == 2
    assert preservation["shared_parameters"] == 3
    assert preservation["maximum_absolute_difference"] == 0.0

    metadata = _metadata(checkpoint)
    assert metadata["selected"]["epoch"] == 3
    assert metadata["selected"]["num_optimizer_updates"] == 12

    summary = _parameter_summary(torch.tensor([-2.0, 0.0, 1.0]))
    assert summary["shape"] == [3]
    assert summary["mean_absolute"] == 1.0
    assert summary["negative_fraction"] == pytest.approx(1 / 3)
    assert summary["near_zero_fraction"] == pytest.approx(1 / 3)
