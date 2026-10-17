# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Identity-started affine adapter for normalized model forcings."""

import torch
from torch import nn


class ModelForcingAdapter(nn.Module):
    """Mix three forcing channels pointwise; restore zero-filled model land."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(3))
        self.bias = nn.Parameter(torch.zeros(3))

    def forward(self, forcing, wet):
        # Float32 identity is exact, including under the surrounding BF16 context.
        with torch.autocast(forcing.device.type, enabled=False):
            output = torch.einsum(
                "oc,btchw->btohw", self.weight.float(), forcing.float()
            )
            output = output + self.bias[None, None, :, None, None]
            return torch.where(wet[None, None, None].bool(), output, 0)
