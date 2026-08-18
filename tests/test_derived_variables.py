# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import torch

from samudra.aggregator.metrics import area_weighted_sum
from samudra.derived_variables import compute_global_ocean_heat_content


def test_ocean_heat_content():
    wet_mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    temperature = torch.ones([1, 3, *wet_mask.shape])
    temperature = temperature * wet_mask
    dz = torch.tensor([1.0, 2.0, 4.0])
    area_weights = torch.ones(wet_mask.shape)
    area_weighted_func = partial(area_weighted_sum, area_weights=area_weights)

    global_heat_content = compute_global_ocean_heat_content(
        temperature,
        dz,
        area_weighted_func,
    )

    assert global_heat_content == 86766120
