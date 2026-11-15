# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, assert_never

import numpy as np
import torch
from torch import nn

from samudra.models.modules.blocks import (
    AvgPool,
    BilinearUpsample,
    CoreBlock,
    CoreBlockBuilder,
    DropPath,
    MaxPool,
    TransposedConvUpsample,
    UpsamplingBlockBuilder,
    ZonallyPeriodicBilinearUpsample,
)
from samudra.models.modules.land import coarsen_mask
from samudra.utils.train import pairwise

if TYPE_CHECKING:
    from samudra.config import Checkpointing  # noqa: F401


class UNetBackbone(nn.Module):
    """A configurable, convolutional or ConvNeXt[1] U-Net[2] implementation.

    Args:
        ch_width (list[int]): The widths of CNN input channels going down into the U-Net. This module first builds
          downsampling CNN blocks before reversing the `ch_widths` to build upsampling CNN blocks. Typically, these
          values should be set in monotonically non-decreasing sizes.
        dilation (list[int]): List of dilation sizes for CNN blocks. See [3] for a general background. This list must
          be one less than the length of `ch_widths`.
        n_layers (list[int]): List of the number of CNN layers to be used in each block section of the U-Net. Typically,
          this is set to all 1s. This value must match the length of `dilation`.
        pad (str): The type of padding to use in all CNN blocks. Passed into `torch.functional.pad`'s `mode` argument.
        create_block: A factory method that creates the CoreBlocks for all CNN layers.
        downsampling_block (nn.Module): A block that downsamples during the descent of the U-Net.
        create_upsampling_block: A factory method that creates upsampling blocks during the ascent of the U-Net.
        checkpointing (Checkpointing | None): The current mode for checkpointing (typically "all" or "simple"). None
          turns checkpointing off.

    References:
        [1]: https://arxiv.org/abs/2201.03545
        [2]: https://arxiv.org/abs/1505.04597
        [3]: https://github.com/vdumoulin/conv_arithmetic/#dilated-convolution-animations.
    """

    def __init__(
        self,
        in_channels: int,
        ch_width: list[int],
        dilation: list[int],
        n_layers: list[int],
        pad: str,
        create_block: CoreBlockBuilder,
        downsampling_block: nn.Module,
        create_upsampling_block: UpsamplingBlockBuilder,
        checkpointing: "Checkpointing | None",
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels: int = ch_width[0]

        # Create local copies of config lists that will be reversed
        ch_width = [in_channels] + ch_width.copy()
        dilation = dilation.copy()
        n_layers = n_layers.copy()
        self.pad = pad

        match checkpointing:
            case "all":
                self.checkpoint_all = True
                checkpoint_simple = False
            case "simple":
                self.checkpoint_all = False
                checkpoint_simple = True
            case None:
                self.checkpoint_all = False
                checkpoint_simple = False
            case _:
                assert_never(checkpointing)

        # going down
        layers: list[nn.Module] = []
        for i, (a, b) in enumerate(pairwise(ch_width)):
            # Core block
            layers.append(
                create_block(
                    in_channels=a,
                    out_channels=b,
                    dilation=dilation[i],
                    n_layers=n_layers[i],
                    pad=pad,
                    checkpoint_simple=checkpoint_simple,
                )
            )
            # Down sampling block
            layers.append(downsampling_block)

        # Middle block
        layers.append(
            create_block(
                in_channels=b,
                out_channels=b,
                dilation=dilation[i],
                n_layers=n_layers[i],
                pad=pad,
                checkpoint_simple=checkpoint_simple,
            )
        )

        # First upsampling
        layers.append(create_upsampling_block(in_channels=b, out_channels=b))

        # Reverse for upsampling path
        ch_width.reverse()
        dilation.reverse()
        n_layers.reverse()

        # going up
        for i, (a, b) in enumerate(pairwise(ch_width[:-1])):
            layers.append(
                create_block(
                    in_channels=a,
                    out_channels=b,
                    dilation=dilation[i],
                    n_layers=n_layers[i],
                    pad=pad,
                    checkpoint_simple=checkpoint_simple,
                )
            )
            layers.append(create_upsampling_block(in_channels=b, out_channels=b))

        # Final conv block
        layers.append(
            create_block(
                in_channels=b,
                out_channels=b,  # this is the same as self.out_channels
                dilation=dilation[i],
                n_layers=n_layers[i],
                pad=pad,
                checkpoint_simple=checkpoint_simple,
            )
        )

        first_block = layers[0]
        assert isinstance(first_block, CoreBlock)
        self.N_pad = first_block.N_pad
        self.layers = nn.ModuleList(layers)
        self.num_steps = int(len(ch_width) - 1)
        self.drop_path = DropPath(drop_path_rate)

    def forward(
        self,
        fts: torch.Tensor,
        mask: torch.Tensor | None = None,
        input_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the U-Net.

        Args:
            fts: Input features, ``(B, C, H, W)``.
            mask: Optional ocean mask, ``(1, 1, H, W)``, 1 on ocean and 0 on land.
                When given, land is absent from every layer (``land: omit``),
                and the output is zero on land. Coarser levels use the max-pooled
                mask: a coarse cell is ocean if any of its fine cells is.
            input_mask: Optional per-channel ocean mask of the input,
                ``(1, C, H, W)``, for the first layer. Defaults to ``mask``.
        """
        if input_mask is not None and mask is None:
            raise ValueError("input_mask requires mask")
        skip_inputs: list[torch.Tensor] = []
        for i in range(self.num_steps):
            skip_inputs.append(torch.zeros_like(fts))
        # masks[k] is the ocean mask at U-Net level k (0 is the input grid).
        masks: list[torch.Tensor] = [] if mask is None else [mask.to(fts.dtype)]
        level = 0
        count = 0
        for layer in self.layers:
            # Circular/Globe padding
            if isinstance(layer, nn.Conv2d):
                fts = torch.nn.functional.pad(
                    fts, (self.N_pad, self.N_pad, 0, 0), mode=self.pad
                )
                fts = torch.nn.functional.pad(
                    fts, (0, 0, self.N_pad, self.N_pad), mode="constant"
                )

            is_upsample = isinstance(
                layer,
                (
                    BilinearUpsample,
                    TransposedConvUpsample,
                    ZonallyPeriodicBilinearUpsample,
                ),
            )
            layer_args: tuple[torch.Tensor, ...] = (fts,)
            if masks:
                layer_args = (fts, masks[level])
                if layer is self.layers[0] and input_mask is not None:
                    layer_args = (fts, masks[level], input_mask)
            # (Maybe) apply checkpointing
            if self.checkpoint_all:
                fts = torch.utils.checkpoint.checkpoint(
                    layer, *layer_args, use_reentrant=False
                )  # type: ignore
            else:
                fts = layer(*layer_args)

            if masks:
                if isinstance(layer, (AvgPool, MaxPool)):
                    level += 1
                    if level == len(masks):
                        masks.append(coarsen_mask(masks[-1], layer.pooling))
                elif is_upsample:
                    level -= 1

            # UNet residuals logic (skip connections)
            if count < self.num_steps:
                if isinstance(layer, CoreBlock):
                    skip_inputs[count] = fts
                    count += 1
            elif count >= self.num_steps:
                if is_upsample:
                    crop = np.array(fts.shape[2:])
                    shape = np.array(
                        skip_inputs[int(2 * self.num_steps - count - 1)].shape[2:]
                    )
                    pads = shape - crop
                    pads = [
                        pads[1] // 2,
                        pads[1] - pads[1] // 2,
                        pads[0] // 2,
                        pads[0] - pads[0] // 2,
                    ]
                    fts = nn.functional.pad(fts, pads)
                    if masks:
                        fts = fts * masks[level].to(fts.dtype)
                    fts += self.drop_path(
                        skip_inputs[int(2 * self.num_steps - count - 1)]
                    )
                    count += 1

        return fts
