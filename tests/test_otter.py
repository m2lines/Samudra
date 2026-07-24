# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from samudra.datasets import TrainData
from samudra.models.modules.otter import OtterBackbone
from samudra.models.otter import Otter
from samudra.utils.ctx import GridContext


def make_context(out_channels: int, height: int, width: int) -> GridContext:
    mask = torch.ones(out_channels, height, width, dtype=torch.bool)
    latitude = torch.linspace(-80, 80, height)
    longitude = torch.linspace(0, 360 - 360 / width, width)
    resolution = (latitude, longitude)
    return GridContext(mask, resolution, resolution)


def make_model(
    *,
    pred_residuals: bool = False,
    checkpoint_blocks: bool = False,
    drop_path_rate: float = 0.0,
) -> Otter:
    in_channels = 10
    out_channels = 6
    backbone = OtterBackbone(
        in_channels=in_channels,
        out_channels=out_channels,
        token_dim=32,
        stage_depths=(1, 1),
        num_heads=4,
        window_size=2,
        patch_size=2,
        position_features=16,
        position_min_scale=0.1,
        position_max_scale=720.0,
        hidden_ratio=2.0,
        ffn_align_to=8,
        dropout_rate=0.0,
        drop_path_rate=drop_path_rate,
        checkpoint_blocks=checkpoint_blocks,
    )
    return Otter(
        in_channels=in_channels,
        out_channels=out_channels,
        pred_residuals=pred_residuals,
        last_kernel_size=1,
        pad="circular",
        backbone=backbone,
        hist=0,
        gradient_detach_interval=0,
        use_bfloat16=False,
    )


def test_forward_supports_padding_and_masks_land():
    model = make_model()
    height, width = 9, 13
    prognostic = torch.randn(2, 6, height, width)
    boundary = torch.randn(2, 4, height, width)
    ctx = make_context(6, height, width)
    ctx.label_mask[:, 2, 3] = False

    output = model.forward_once(prognostic, boundary, ctx)

    assert output.shape == prognostic.shape
    assert torch.equal(output[:, :, 2, 3], torch.zeros_like(output[:, :, 2, 3]))


@pytest.mark.parametrize("checkpoint_blocks", [False, True])
def test_backward_reaches_input(checkpoint_blocks):
    model = make_model(checkpoint_blocks=checkpoint_blocks)
    model.train()
    prognostic = torch.randn(2, 6, 8, 8, requires_grad=True)
    boundary = torch.randn(2, 4, 8, 8, requires_grad=True)

    output = model.forward_once(prognostic, boundary, make_context(6, 8, 8))
    output.square().mean().backward()

    assert prognostic.grad is not None
    assert boundary.grad is not None
    assert torch.isfinite(prognostic.grad).all()
    assert torch.isfinite(boundary.grad).all()


def test_existing_blockwise_residual_contract_is_unchanged():
    model = make_model(pred_residuals=True)
    model.eval()
    prognostic = torch.randn(2, 6, 8, 8)
    boundary = torch.randn(2, 4, 8, 8)
    label = torch.randn_like(prognostic)
    ctx = make_context(6, 8, 8)
    batch = TrainData(
        num_prognostic_channels=6,
        num_boundary_channels=4,
        ctx=ctx,
    )
    batch.append(prognostic, boundary, label)

    with torch.no_grad():
        decoding = model.forward_once(prognostic, boundary, ctx)
        prediction = model(batch)[0]

    assert torch.allclose(prediction, prognostic + decoding)


def test_dropped_blocks_keep_parameters_in_gradient_graph(monkeypatch):
    model = make_model(drop_path_rate=0.5)
    model.train()
    monkeypatch.setattr(
        torch,
        "rand",
        lambda *size, **kwargs: torch.zeros(*size, **kwargs),
    )
    prognostic = torch.randn(1, 6, 8, 8)
    boundary = torch.randn(1, 4, 8, 8)

    output = model.forward_once(prognostic, boundary, make_context(6, 8, 8))
    output.mean().backward()

    assert all(parameter.grad is not None for parameter in model.parameters())


def test_resolution_mismatch_raises():
    model = make_model()
    with pytest.raises(ValueError, match="resolution size mismatch"):
        model.forward_once(
            torch.randn(1, 6, 8, 8),
            torch.randn(1, 4, 8, 8),
            make_context(6, 7, 8),
        )
