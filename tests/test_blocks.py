import pytest
import torch
import torch.nn as nn
from hypothesis import given, settings
from hypothesis import strategies as st

from ocean_emulators.models.modules.blocks import (
    ConvNeXtBlock,
    DropPath,
    PointwiseLinear,
    RepConvNeXtBlock,
    TrueConvNeXtBlock,
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


class TestTrueConvNeXtBlock:
    @pytest.mark.parametrize("kernel_size", [3, 7, 15])
    def test_preserves_shape_and_passes_gradients(self, kernel_size: int):
        block = TrueConvNeXtBlock(
            in_channels=8, out_channels=12, kernel_size=kernel_size
        )
        x = torch.randn(2, 8, 16, 24, requires_grad=True)
        y = block(x)
        assert y.shape == (2, 12, 16, 24)
        y.sum().backward()
        assert x.grad is not None and x.grad.shape == x.shape

    @pytest.mark.parametrize("kernel_size", [3, 7, 15])
    def test_zonal_translation_equivariance(self, kernel_size: int):
        """Output shifts with the input along longitude (W axis).

        Verifies the circular-x / zero-y padding is plumbed correctly: the
        block must be exactly equivariant to longitudinal shifts. Eval mode
        keeps BatchNorm using fixed running stats so the only source of
        non-equivariance would be a misapplied padding mode.
        """
        torch.manual_seed(0)
        block = TrueConvNeXtBlock(
            in_channels=8, out_channels=8, kernel_size=kernel_size, norm="batch"
        ).eval()

        x = torch.randn(2, 8, 12, 24)
        shift = 5

        with torch.no_grad():
            y = block(x)
            y_shifted_input = block(torch.roll(x, shifts=shift, dims=-1))

        torch.testing.assert_close(
            y_shifted_input, torch.roll(y, shifts=shift, dims=-1)
        )


class TestRepConvNeXtBlock:
    @pytest.mark.parametrize("kernel_size", [7, 15])
    def test_fold_reparam_preserves_output(self, kernel_size: int):
        """Folding Conv+BN per branch and merging into a single conv must
        produce a numerically identical forward pass — the contract that
        lets us train with parallel branches and deploy the merged conv.

        BN running stats need to be populated for fold to be meaningful;
        we run a dummy training-mode forward first, then evaluate in eval
        mode so the running stats (not batch stats) are used.
        """
        torch.manual_seed(0)
        block = RepConvNeXtBlock(
            in_channels=8, out_channels=8, kernel_size=kernel_size, norm="batch"
        )

        # Populate BN running stats with a few train-mode forwards.
        block.train()
        for _ in range(3):
            with torch.no_grad():
                block(torch.randn(2, 8, 12, 24))

        block.eval()
        x = torch.randn(2, 8, 12, 24)
        with torch.no_grad():
            y_before = block(x)
            block.fold_reparam()
            assert block.dwconv_small is None
            assert block.bn_small is None
            assert isinstance(block.bn_large, nn.Identity)
            y_after = block(x)

        torch.testing.assert_close(y_before, y_after, rtol=1e-5, atol=1e-6)

    def test_fold_reparam_idempotent(self):
        """Calling fold_reparam twice is a no-op on the second call."""
        block = RepConvNeXtBlock(
            in_channels=4, out_channels=4, kernel_size=7, norm="batch"
        ).eval()
        block.fold_reparam()
        # Second call must not error and must leave the merged conv intact.
        before_w = block.dwconv.weight.data.clone()
        block.fold_reparam()
        torch.testing.assert_close(block.dwconv.weight.data, before_w)

    def test_no_small_branch_for_kernel_3(self):
        """When kernel_size == SMALL_KERNEL the parallel branch is skipped:
        adding a 3×3 next to a 3×3 main conv would be redundant."""
        block = RepConvNeXtBlock(
            in_channels=4, out_channels=4, kernel_size=3, norm="batch"
        )
        assert block.dwconv_small is None
        assert block.bn_small is None

    def test_requires_batch_norm(self):
        """Fold relies on BN folding identity; assert other norms are rejected."""
        with pytest.raises(AssertionError, match="norm='batch'"):
            RepConvNeXtBlock(in_channels=4, out_channels=4, kernel_size=7, norm="layer")


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
