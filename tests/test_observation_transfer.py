# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from samudra.experiments import initializer_models
from samudra.experiments.observation_metrics import PROTOCOL, selection_score


def test_per_frame_validity_retains_old_path_and_surface_copy(monkeypatch):
    monkeypatch.setattr(
        initializer_models, "make_unet", lambda i, o, w: nn.Conv2d(i, o, 1)
    )
    model = initializer_models.HistoryInitializer(
        ["thetao_0", "so_0", "zos"], "wide", True
    )
    surface = torch.randn(1, 19, 2, 8, 8)
    forcing = torch.randn(1, 19, 3, 8, 8)
    context = torch.randn(1, 5, 8, 8)
    mask = torch.ones(3, 8, 8)
    old = model(surface, forcing, context, mask)
    valid = torch.ones_like(surface)
    new = model(surface, forcing, context, mask, valid)
    torch.testing.assert_close(old, new, rtol=0, atol=0)
    valid[:, 3, :, 4, 4] = 0
    missing = model(surface, forcing, context, mask, valid)
    torch.testing.assert_close(missing[:, :, [0, 2]], surface[:, -2:])
    assert not torch.equal(missing[:, :, 1], old[:, :, 1])
    with pytest.raises(ValueError):
        model(surface, forcing, context, mask, valid[:, :, :1])


def test_spectral_collapse_can_lose_despite_improved_pointwise_errors():
    control = {"metrics": {key: 1.0 for key in PROTOCOL["integrated"]}}
    good: dict[str, Any] = {
        "metrics": {key: 1.0 for key in PROTOCOL["integrated"]},
        "spectra": {"sst/open/day30": {"error_dex": 0.05}},
    }
    collapsed: dict[str, Any] = {
        "metrics": {key: 0.8 for key in PROTOCOL["integrated"]},
        "spectra": {"sst/open/day30": {"error_dex": 1.0}},
    }
    keys = list(good["spectra"])
    assert selection_score(good, control, keys) < selection_score(
        collapsed, control, keys
    )
    with pytest.raises(ValueError):
        selection_score(good, control, [])
    collapsed["spectra"][keys[0]]["error_dex"] = np.nan
    with pytest.raises(ValueError):
        selection_score(collapsed, control, keys)
