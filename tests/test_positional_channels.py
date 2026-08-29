# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import torch

from samudra.config import SamudraConfig, UNetBackboneConfig
from samudra.utils.ctx import GridContext
from samudra.utils.data import DataSource


def test_positional_parameters_update(dummy_src: DataSource):
    """Verify that positional parameters can learn something in a tiny example."""
    src = dummy_src
    h, w = src.grid_size
    masks = src.masks

    # Create the model itself with learned positional embeddings
    config = SamudraConfig(
        unet=UNetBackboneConfig(
            ch_width=[2],
            dilation=[1],
            n_layers=[1],
        ),
        pos_channels=1,
    )
    model = config.build(
        prog_channels=1,
        boundary_channels=1,
        out_channels=1,
        hist=0,
        srcs=[src],
    )

    # Verify we have created the positional embeddings
    assert model.positional_params is not None
    assert model.positional_params.shape == (config.pos_channels, h, w)
    assert not torch.allclose(
        model.positional_params.detach(),
        torch.zeros_like(model.positional_params),
    )

    # Run a step and confirm they have changed
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    prog = torch.randn(1, 1, h, w)
    boundary = torch.randn(1, 1, h, w)
    optimizer.zero_grad()
    out = model.forward_once(
        prog,
        boundary,
        GridContext(masks.prognostic, src.resolution, src.resolution),
    )
    loss = out.sum()
    loss.backward()
    before = model.positional_params.detach().clone()
    optimizer.step()
    assert not torch.allclose(model.positional_params.detach(), before)
