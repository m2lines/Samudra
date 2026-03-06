"""Tests for blocks.py, focusing on PointwiseLinear equivalence to Conv2d(kernel_size=1)."""

import pytest
import torch
import torch.nn as nn

from ocean_emulators.models.modules.blocks import ConvNeXtBlock, PointwiseLinear


def test_pointwise_linear_equivalent_to_conv1x1():
    """PointwiseLinear produces the same output as Conv2d(kernel_size=1)
    when initialized with the same weights."""
    pw = PointwiseLinear(16, 32)
    conv = nn.Conv2d(16, 32, kernel_size=1)

    with torch.no_grad():
        conv.weight.copy_(pw.linear.weight.unsqueeze(-1).unsqueeze(-1))
        assert conv.bias is not None and pw.linear.bias is not None
        conv.bias.copy_(pw.linear.bias)

    x = torch.randn(2, 16, 8, 10)
    torch.testing.assert_close(pw(x), conv(x))


def test_conv1x1_equivalent_to_pointwise_linear():
    """Conv2d(kernel_size=1) produces the same output as PointwiseLinear
    when initialized with the same weights (reverse direction)."""
    conv = nn.Conv2d(16, 32, kernel_size=1)
    pw = PointwiseLinear(16, 32)

    with torch.no_grad():
        pw.linear.weight.copy_(conv.weight.squeeze(-1).squeeze(-1))
        assert pw.linear.bias is not None and conv.bias is not None
        pw.linear.bias.copy_(conv.bias)

    x = torch.randn(2, 16, 8, 10)
    torch.testing.assert_close(conv(x), pw(x))


@pytest.mark.parametrize("pointwise_linear", [True, False])
def test_convnext_block_backward_pass(pointwise_linear: bool):
    """Both modes support backward pass without errors."""
    block = ConvNeXtBlock(
        in_channels=16,
        out_channels=32,
        kernel_size=3,
        dilation=1,
        n_layers=1,
        pointwise_linear=pointwise_linear,
    )
    x = torch.randn(2, 16, 8, 10, requires_grad=True)
    y = block(x)
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
