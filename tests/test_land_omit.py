# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``land: omit``, which makes land absent from every U-Net layer."""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from samudra.config import BlockConfig, SamudraConfig, UNetBackboneConfig
from samudra.models.modules.blocks import ZonallyPeriodicBilinearUpsample
from samudra.models.modules.land import (
    LandMode,
    masked_avg_pool2d,
    masked_batch_norm,
    masked_upsample,
    partial_conv2d,
)
from samudra.utils.ctx import BatchGrid

H, W = 16, 32
N_VARS = 2  # prognostic variables; variable 1 is "deeper", with a smaller mask
INPUT_STEPS = 2
BOUNDARY = 2

CONFIGS = {
    "convnext_avgpool_zonal": SamudraConfig(
        unet=UNetBackboneConfig(
            ch_width=[4, 6],
            dilation=[1, 2],
            n_layers=[1, 1],
            core_block=BlockConfig(upscale_factor=2),
        ),
        pos_channels=2,
        add_3d_coordinates=True,
        checkpointing="all",
    ),
    "convblock_maxpool_bilinear": SamudraConfig(
        unet=UNetBackboneConfig(
            ch_width=[4, 6],
            dilation=[1, 2],
            n_layers=[2, 1],
            core_block=BlockConfig(block_type="conv_block"),
            down_sampling_block="max_pool",
            up_sampling_block="bilinear_upsample",
        ),
        pred_residuals=True,
    ),
}


def _nested_masks(generator: torch.Generator) -> torch.Tensor:
    """Per-variable ocean masks that nest with depth, as OM4's do."""
    surface = torch.rand(H, W, generator=generator) > 0.3
    deep = surface & (torch.rand(H, W, generator=generator) > 0.3)
    return torch.stack([surface, deep])


def _build(land: LandMode, name: str, **overrides):
    torch.manual_seed(0)
    config = SamudraConfig.model_validate(
        CONFIGS[name].model_dump() | {"land": land} | overrides
    )
    return config.build(
        prog_channels=N_VARS * INPUT_STEPS,
        boundary_channels=BOUNDARY,
        out_channels=N_VARS,
        input_steps=INPUT_STEPS,
        grid_sizes=[(H, W)],
    )


def _lat_lon():
    lat = torch.linspace(-80, 80, H, dtype=torch.float64)
    lon = torch.linspace(0, 360 - 360 / W, W, dtype=torch.float64)
    return lat, lon


@pytest.mark.parametrize("name", CONFIGS)
@pytest.mark.parametrize("training", [True, False])
def test_forecast_and_gradients_ignore_land_values(name: str, training: bool):
    """With land omitted, any land values (huge, NaN) leave the forecast and,
    in training, the parameter gradients exactly unchanged."""
    g = torch.Generator().manual_seed(1)
    masks = _nested_masks(g)
    ctx = BatchGrid(masks, _lat_lon(), _lat_lon())
    prog = torch.randn(3, N_VARS * INPUT_STEPS, H, W, generator=g)
    boundary = torch.randn(3, BOUNDARY, H, W, generator=g)
    # Land of each channel: its own depth's mask for prognostics, the surface
    # mask for the boundary forcing.
    prog_land = ~masks.repeat(INPUT_STEPS, 1, 1).expand_as(prog)
    boundary_land = ~masks[0].expand_as(boundary)

    def run(prog, boundary):
        model = _build("omit", name)
        model.train(training)
        out = model.forward_once(prog, boundary, ctx)
        if not training:
            return out, []
        out.square().sum().backward()
        return out, [p.grad for p in model.parameters()]

    clean_out, clean_grads = run(prog, boundary)
    corrupt_prog = prog.masked_fill(prog_land, 1e6)
    corrupt_prog[:, 0][prog_land[:, 0]] = float("nan")
    corrupt_out, corrupt_grads = run(
        corrupt_prog, boundary.masked_fill(boundary_land, -1e6)
    )

    torch.testing.assert_close(corrupt_out, clean_out, rtol=0, atol=0)
    for clean, corrupt in zip(clean_grads, corrupt_grads, strict=True):
        torch.testing.assert_close(corrupt, clean, rtol=0, atol=0)


def test_masked_spatial_ops_preserve_constant_fields():
    """Partial convolution, pooling and upsampling renormalize by the ocean
    count, so a constant ocean field stays constant on ocean, even at coasts
    and next to the poles."""
    g = torch.Generator().manual_seed(2)
    mask = (torch.rand(1, 1, H, W, generator=g) > 0.4).double()
    x = torch.where(mask > 0, 3.0, 1e3) * torch.ones(2, 5, H, W, dtype=torch.float64)

    conv = nn.Conv2d(5, 4, 3, dilation=2).double()
    nn.init.constant_(conv.weight, 1.0 / (9 * 5))
    assert conv.bias is not None
    nn.init.zeros_(conv.bias)
    conv_out = partial_conv2d(x, conv, mask, n_pad=2, pad_mode="circular")
    assert torch.allclose(conv_out[:, :, mask[0, 0] > 0], torch.tensor(3.0).double())

    coarse = torch.nn.functional.max_pool2d(mask, 2)
    pooled = masked_avg_pool2d(x, mask, 2)
    assert torch.allclose(pooled[:, :, coarse[0, 0] > 0], torch.tensor(3.0).double())

    coarse_x = torch.where(coarse > 0, 3.0, 1e3).double().expand(2, 5, -1, -1)
    up = masked_upsample(coarse_x, coarse, ZonallyPeriodicBilinearUpsample())
    assert torch.allclose(up[:, :, mask[0, 0] > 0], torch.tensor(3.0).double())

    for out, m in [(conv_out, mask), (pooled, coarse)]:
        assert (out[:, :, m[0, 0] == 0] == 0).all()


def test_masked_batch_norm_matches_batch_norm_over_ocean_cells():
    """In training and evaluation, the masked batch norm equals nn.BatchNorm2d
    applied to the ocean cells alone, including its running statistics."""
    g = torch.Generator().manual_seed(3)
    mask = torch.rand(H, W, generator=g) > 0.4
    masked, reference = nn.BatchNorm2d(4).double(), nn.BatchNorm2d(4).double()
    for bn in (masked, reference):
        nn.init.normal_(bn.weight, generator=g)
        nn.init.normal_(bn.bias, generator=g)
    reference.load_state_dict(masked.state_dict())

    for training in [True, True, False]:
        masked.train(training)
        reference.train(training)
        x = 2 + 3 * torch.randn(3, 4, H, W, generator=g, dtype=torch.float64)
        out = masked_batch_norm(x, masked, mask.view(1, 1, H, W).double())
        expected = reference(x[:, :, mask].unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(out[:, :, mask], expected)
        assert (out[:, :, ~mask] == 0).all()
    for name, buf in reference.named_buffers():
        torch.testing.assert_close(dict(masked.named_buffers())[name], buf)


def _loss_weights(x: torch.Tensor) -> torch.Tensor:
    return torch.arange(x.numel(), dtype=x.dtype).view_as(x).sin()


def _sync_bn_worker(rank: int, world_size: int, port: int, x, mask, result):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        torch.manual_seed(0)
        bn = nn.SyncBatchNorm.convert_sync_batchnorm(
            nn.Sequential(nn.BatchNorm2d(x.shape[1]))
        )[0].double()
        local = x.chunk(world_size)[rank].clone().requires_grad_(True)
        out = masked_batch_norm(local, bn, mask)
        (out * _loss_weights(x).chunk(world_size)[rank]).sum().backward()
        result[rank] = (out.detach(), local.grad, bn.running_mean, bn.running_var)
    finally:
        dist.destroy_process_group()


def test_masked_sync_batch_norm_matches_single_process_on_global_batch():
    """Under DDP, train.py converts batch norm to SyncBatchNorm. The masked
    version then uses ocean statistics over the global batch, and its
    gradients match a single process that sees the whole batch."""
    g = torch.Generator().manual_seed(4)
    x = torch.randn(4, 3, H, W, generator=g, dtype=torch.float64)
    mask = (torch.rand(1, 1, H, W, generator=g) > 0.4).double()

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    result = mp.Manager().dict()
    mp.spawn(_sync_bn_worker, args=(2, port, x, mask, result), nprocs=2)

    torch.manual_seed(0)
    bn = nn.BatchNorm2d(3).double()
    full = x.clone().requires_grad_(True)
    out = masked_batch_norm(full, bn, mask)
    (out * _loss_weights(x)).sum().backward()

    torch.testing.assert_close(torch.cat([result[r][0] for r in range(2)]), out)
    torch.testing.assert_close(torch.cat([result[r][1] for r in range(2)]), full.grad)
    for r in range(2):
        torch.testing.assert_close(result[r][2], bn.running_mean)
        torch.testing.assert_close(result[r][3], bn.running_var)


@pytest.mark.parametrize("name", CONFIGS)
def test_checkpoint_loads_with_either_land_setting(name: str):
    """land: omit has exactly the parameters and buffers of land: mask."""
    omit = _build("omit", name)
    omit.load_state_dict(_build("mask", name).state_dict(), strict=True)


def test_longitude_roll_equivariance():
    """Rolling inputs and masks in longitude by a multiple of the total
    downsampling factor rolls the forecast the same way."""
    model = _build(
        "omit", "convnext_avgpool_zonal", pos_channels=0, add_3d_coordinates=False
    ).eval()
    g = torch.Generator().manual_seed(5)
    masks = _nested_masks(g)
    prog = torch.randn(1, N_VARS * INPUT_STEPS, H, W, generator=g)
    boundary = torch.randn(1, BOUNDARY, H, W, generator=g)
    shift = 8  # two levels of 2x pooling → multiples of 4

    def forecast(prog, boundary, masks):
        return model.forward_once(
            prog, boundary, BatchGrid(masks, _lat_lon(), _lat_lon())
        )

    rolled = forecast(
        prog.roll(shift, -1), boundary.roll(shift, -1), masks.roll(shift, -1)
    )
    torch.testing.assert_close(rolled, forecast(prog, boundary, masks).roll(shift, -1))


def test_omit_rejects_unsupported_upsampling():
    with pytest.raises(ValueError, match="transposed_conv"):
        SamudraConfig(
            land="omit", unet=UNetBackboneConfig(up_sampling_block="transposed_conv")
        )
