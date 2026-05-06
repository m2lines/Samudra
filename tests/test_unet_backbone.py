# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

from ocean_emulators.models.modules.blocks import (
    AvgPool,
    ConvNeXtBlock,
    ZonallyPeriodicBilinearUpsample,
)
from ocean_emulators.models.modules.unet_backbone import UNetBackbone


def _build_tiny_backbone(drop_path_rate: float = 0.5) -> UNetBackbone:
    """Smallest possible UNet that still has multiple skip connections."""

    def create_block(
        in_channels: int,
        out_channels: int,
        dilation: int,
        n_layers: int,
        pad: str,
        checkpoint_simple: bool,
    ):
        return ConvNeXtBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            dilation=dilation,
            n_layers=1,
            pad=pad,
            upscale_factor=1,
        )

    def create_upsample(in_channels: int, out_channels: int) -> nn.Module:
        return ZonallyPeriodicBilinearUpsample()

    return UNetBackbone(
        in_channels=4,
        ch_width=[8, 12],
        dilation=[1, 2],
        n_layers=[1, 1],
        pad="circular",
        create_block=create_block,
        downsampling_block=AvgPool(),
        create_upsampling_block=create_upsample,
        checkpointing=None,
        drop_path_rate=drop_path_rate,
    )


def test_with_skip_mask_drops_change_output_and_restore_state():
    """The mask context manager actually injects the requested mask, then
    restores prior state on exit.

    Three orthogonal axes here:
      1. all-False mask = identity = unmasked-eval prediction (no-op when
         every skip is kept).
      2. all-True mask zeroes every skip, producing a different prediction
         (proves the mask is wired through to DropPath).
      3. The context manager restores the prior mask on exit, including
         under nesting.
    """
    torch.manual_seed(0)
    backbone = _build_tiny_backbone()
    backbone.eval()
    x = torch.randn(2, 4, 16, 16)

    n_skips = backbone.num_steps
    keep_all = (False,) * n_skips
    drop_all = (True,) * n_skips

    baseline = backbone(x)

    with backbone.with_skip_mask(keep_all):
        with_keep = backbone(x)
    with backbone.with_skip_mask(drop_all):
        with_drop = backbone(x)

    # all-False keep: identical to unmasked eval (DropPath in eval mode is
    # already a no-op, so this just confirms the explicit-keep path matches).
    torch.testing.assert_close(with_keep, baseline)
    # all-True drops: predictions must differ — the bottleneck-only path is a
    # different function from the multi-scale path.
    assert not torch.allclose(with_drop, baseline)

    # State restored after context exits.
    assert backbone._skip_mask is None

    # Nested contexts restore the outer mask.
    with backbone.with_skip_mask(drop_all):
        with backbone.with_skip_mask(keep_all):
            assert backbone._skip_mask == keep_all
        assert backbone._skip_mask == drop_all
    assert backbone._skip_mask is None


def test_with_skip_mask_rejects_wrong_length():
    backbone = _build_tiny_backbone()
    bad_mask = (False,) * (backbone.num_steps + 1)
    with pytest.raises(ValueError, match="skip mask length"):
        with backbone.with_skip_mask(bad_mask):
            pass
