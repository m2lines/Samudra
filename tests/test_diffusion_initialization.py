# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

from samudra.experiments.diffusion_initialization import load_pretraining_source


def test_scratch_imports_only_dynamics_and_keeps_fresh_encoder_and_adapter():
    core = nn.Module()
    core.initializer = nn.Linear(3, 4)
    core.evolution = nn.Linear(4, 4)
    core.adapter = nn.Linear(2, 3)
    before = {k: v.clone() for k, v in core.state_dict().items()}
    source = {
        k: torch.full_like(v, 42)
        for k, v in before.items()
        if not k.startswith("adapter.")
    }
    load_pretraining_source(core, source, initialization="scratch")
    for name, value in core.state_dict().items():
        torch.testing.assert_close(
            value, source[name] if name.startswith("evolution.") else before[name]
        )
    with pytest.raises(RuntimeError):
        load_pretraining_source(core, {}, initialization="scratch")
