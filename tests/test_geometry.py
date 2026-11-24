# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from samudra.config import BlockConfig, SamudraConfig, UNetBackboneConfig
from samudra.models.modules.geometry import gaussian_geometry
from samudra.utils.ctx import BatchGrid


def coordinates(h=36, w=72):
    lat = np.rad2deg(np.arcsin(np.polynomial.legendre.leggauss(h)[0]))
    return torch.tensor(lat), torch.arange(w, dtype=torch.float64) * (360 / w) + 180 / w


def model(mode, checkpointing=None):
    torch.manual_seed(15)
    return SamudraConfig(
        checkpointing=checkpointing,
        unet=UNetBackboneConfig(
            ch_width=[8, 12],
            dilation=[1, 1],
            n_layers=[1, 1],
            geometry_mode=mode,
            core_block=BlockConfig(
                block_type="conv_next_block", norm="instance", upscale_factor=2
            ),
        ),
    ).build(3, 1, 3, 1, [(36, 72)])


@pytest.mark.parametrize("checkpointing", [None, "all"])
def test_identity_initialization_and_gradients(checkpointing):
    baseline = model("none", checkpointing)
    rng = torch.get_rng_state()
    x, b = torch.randn(1, 3, 36, 72), torch.randn(1, 1, 36, 72)
    ctx = BatchGrid(
        torch.ones(3, 36, 72, dtype=torch.bool), coordinates(), coordinates()
    )
    expected = baseline.forward_once(x, b, ctx)
    for mode in ["concat", "film"]:
        candidate = model(mode, checkpointing)
        assert torch.equal(rng, torch.get_rng_state())
        actual = candidate.forward_once(x, b, ctx)
        torch.testing.assert_close(actual, expected, rtol=0, atol=2e-6)
        actual.square().mean().backward()
        if mode == "film":
            heads = candidate.unet.geometry.heads
            assert all(head.weight.grad.abs().sum() > 0 for head in heads)
            # The zero head delays encoder gradients by one optimizer step.
            torch.optim.SGD(candidate.parameters(), lr=0.01).step()
            candidate.zero_grad()
            candidate.forward_once(x, b, ctx).square().mean().backward()
            assert candidate.unet.geometry.encoder[0].weight.grad.abs().sum() > 0
        else:
            first = candidate.unet.layers[0]
            assert first.convblock[0].weight.grad[:, -2:].abs().sum() > 0
            assert first.skip_module.weight.grad[:, -2:].abs().sum() > 0


def test_geometry_footprints_and_reference():
    lat, lon = coordinates(180, 360)
    maps = gaussian_geometry(lat, lon, 5)
    assert [g.shape[-2:] for g in maps] == [
        (180, 360),
        (90, 180),
        (45, 90),
        (22, 45),
        (11, 22),
    ]
    # Fine dy integrates from pole to pole; coarser levels crop odd trailing cells.
    dy = maps[0][0, 1, :, 0].exp().double() * 100_000
    assert dy.sum().item() == pytest.approx(np.pi * 6_371_000, rel=1e-6)
    assert maps[0][0, 0, 90, 0] > maps[0][0, 0, 0, 0]
    shifted = gaussian_geometry(lat, lon, 5, 200_000)
    for a, b in zip(maps, shifted):
        torch.testing.assert_close(a - b, torch.full_like(a, np.log(2)))
    for fine, coarse in zip(maps, maps[1:]):
        fine_dy = fine[0, 1, :, 0].exp() * 100_000
        coarse_dy = coarse[0, 1, :, 0].exp() * 100_000
        torch.testing.assert_close(
            coarse_dy, fine_dy[: 2 * len(coarse_dy)].reshape(-1, 2).sum(1)
        )


def test_reject_non_gaussian_coordinates():
    with pytest.raises(ValueError, match="Gaussian"):
        gaussian_geometry(torch.linspace(-89, 89, 36), coordinates()[1], 3)


def test_odd_grid_forward_and_round_trip():
    cfg = SamudraConfig(
        unet=UNetBackboneConfig(
            ch_width=[4, 6, 8, 10],
            dilation=[1] * 4,
            n_layers=[1] * 4,
            geometry_mode="film",
            core_block=BlockConfig(block_type="conv_next_block", norm="instance"),
        )
    )
    net = cfg.build(3, 1, 3, 1, [(180, 360)])
    ctx = BatchGrid(
        torch.ones(3, 180, 360, dtype=torch.bool),
        coordinates(180, 360),
        coordinates(180, 360),
    )
    x, b = torch.randn(1, 3, 180, 360), torch.randn(1, 1, 180, 360)
    result = net.forward_once(x, b, ctx)
    restored = cfg.build(3, 1, 3, 1, [(180, 360)])
    restored.load_state_dict(net.state_dict(), strict=True)
    torch.testing.assert_close(restored.forward_once(x, b, ctx), result)
