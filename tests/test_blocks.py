# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn
from hypothesis import given, settings
from hypothesis import strategies as st

from ocean_emulators.models.modules.blocks import (
    ConvNeXtBlock,
    DropPath,
    PointwiseLinear,
)


@given(
    in_ch=st.integers(min_value=1, max_value=64),
    out_ch=st.integers(min_value=1, max_value=64),
    batch=st.integers(min_value=1, max_value=4),
    height=st.integers(min_value=1, max_value=16),
    width=st.integers(min_value=1, max_value=16),
)
@settings(max_examples=20)
def test_pointwise_linear_equivalent_to_conv1x1(
    in_ch: int, out_ch: int, batch: int, height: int, width: int
):
    """PointwiseLinear produces the same output as Conv2d(kernel_size=1)
    for arbitrary channel counts and spatial dimensions."""
    pw = PointwiseLinear(in_ch, out_ch)
    conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    with torch.no_grad():
        conv.weight.copy_(pw.linear.weight.unsqueeze(-1).unsqueeze(-1))
        assert conv.bias is not None and pw.linear.bias is not None
        conv.bias.copy_(pw.linear.bias)

    x = torch.randn(batch, in_ch, height, width)
    torch.testing.assert_close(pw(x), conv(x))


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


class TestDropPath:
    def test_no_op_during_eval(self):
        """During eval, skip connections always pass through regardless of drop_prob."""
        drop_path = DropPath(drop_prob=1.0)
        drop_path.eval()
        skip = torch.randn(4, 16, 8, 8)
        torch.testing.assert_close(drop_path(skip), skip)

    def test_drop_prob_one_zeros_all_skip_connections(self):
        """With drop_prob=1.0, all skip connections are zeroed out,
        forcing the model to rely entirely on the trunk (deep path)."""
        drop_path = DropPath(drop_prob=1.0)
        drop_path.train()

        trunk = torch.randn(4, 16, 8, 8)
        skip = torch.randn(4, 16, 8, 8)

        output = trunk + drop_path(skip)
        torch.testing.assert_close(output, trunk)

    def test_gradient_flows_through_trunk_when_skip_dropped(self):
        """When skip connections are dropped, gradients must still flow
        through the trunk (the deep/bottleneck path)."""
        drop_path = DropPath(drop_prob=1.0)
        drop_path.train()

        trunk = torch.randn(2, 16, 8, 8, requires_grad=True)
        skip = torch.randn(2, 16, 8, 8, requires_grad=True)

        output = trunk + drop_path(skip)
        output.sum().backward()

        assert trunk.grad is not None, "Gradient must flow through the trunk."
        assert skip.grad is not None, "Skip should still get a (zero) gradient."
        # Trunk gradient should be all ones (d/d(trunk) of sum(trunk + 0))
        torch.testing.assert_close(trunk.grad, torch.ones_like(trunk))
        # Skip gradient should be all zeros (dropped)
        torch.testing.assert_close(skip.grad, torch.zeros_like(skip))

    def test_set_progress_linearly_decays_rate(self):
        """set_progress linearly interpolates dropout.p from max_drop_prob to 0."""
        drop_path = DropPath(drop_prob=0.6)

        drop_path.set_progress(0.0)
        assert drop_path.dropout.p == pytest.approx(0.6)

        drop_path.set_progress(0.5)
        assert drop_path.dropout.p == pytest.approx(0.3)

        drop_path.set_progress(1.0)
        assert drop_path.dropout.p == pytest.approx(0.0)

    def test_set_progress_beyond_one_keeps_drop_path_off(self):
        """Once the early phase is over (progress >= 1), drop path stays disabled."""
        drop_path = DropPath(drop_prob=0.5)
        drop_path.train()

        drop_path.set_progress(2.0)
        assert drop_path.dropout.p == pytest.approx(0.0)

        # Confirm it actually passes through (not just p==0 but forward is identity)
        skip = torch.randn(4, 16, 8, 8)
        torch.testing.assert_close(drop_path(skip), skip)
