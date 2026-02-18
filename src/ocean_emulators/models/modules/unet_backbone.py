from typing import TYPE_CHECKING, assert_never

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ocean_emulators.models.modules.blocks import (
    BilinearUpsample,
    CoreBlock,
    CoreBlockBuilder,
    TransposedConvUpsample,
    UpsamplingBlockBuilder,
    ZonallyPeriodicBilinearUpsample,
)
from ocean_emulators.utils.train import pairwise

if TYPE_CHECKING:
    from ocean_emulators.config import Checkpointing  # noqa: F401


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
        *,
        pos_channels: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels: int = ch_width[0]
        self.pos_channels = pos_channels

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
        encoder_in_channels: list[int] = []
        for i, (a, b) in enumerate(pairwise(ch_width)):
            encoder_in_channels.append(a)
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

        # Optional learned positional injection: add a projected, downsampled pos map
        # to the activations at each encoder CoreBlock input, plus the bottleneck.
        if pos_channels > 0:
            # One per encoder CoreBlock, plus one for the bottleneck CoreBlock.
            pos_proj_out_channels = encoder_in_channels + [b]
            self.pos_projs: nn.ModuleList | None = nn.ModuleList(
                [
                    nn.Conv2d(pos_channels, c, kernel_size=1, bias=False)
                    for c in pos_proj_out_channels
                ]
            )
            # Keep scale at 1.0; positional_params are initialized very small.
            self.pos_scales = nn.Parameter(torch.ones(len(self.pos_projs)))
        else:
            self.pos_projs = None
            self.register_parameter("pos_scales", None)

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

    def _inject_pos(
        self, fts: torch.Tensor, pos: torch.Tensor, idx: int
    ) -> torch.Tensor:
        assert self.pos_projs is not None
        assert self.pos_scales is not None
        pos_rs = F.interpolate(
            pos,
            size=fts.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).to(device=fts.device, dtype=fts.dtype)
        return fts + self.pos_scales[idx] * self.pos_projs[idx](pos_rs)

    def forward(
        self, fts: torch.Tensor, *, pos: torch.Tensor | None = None
    ) -> torch.Tensor:
        skip_inputs: list[torch.Tensor] = []
        for i in range(self.num_steps):
            skip_inputs.append(torch.zeros_like(fts))
        count = 0
        injected_bottleneck = False
        for layer in self.layers:
            if (
                pos is not None
                and self.pos_projs is not None
                and isinstance(layer, CoreBlock)
            ):
                # Encoder: inject before each encoder CoreBlock (count = stage index).
                if count < self.num_steps:
                    fts = self._inject_pos(fts, pos, idx=count)
                # Bottleneck: inject before the middle CoreBlock exactly once.
                elif count == self.num_steps and not injected_bottleneck:
                    fts = self._inject_pos(fts, pos, idx=self.num_steps)
                    injected_bottleneck = True

            # Circular/Globe padding
            if isinstance(layer, nn.Conv2d):
                fts = torch.nn.functional.pad(
                    fts, (self.N_pad, self.N_pad, 0, 0), mode=self.pad
                )
                fts = torch.nn.functional.pad(
                    fts, (0, 0, self.N_pad, self.N_pad), mode="constant"
                )

            # (Maybe) apply checkpointing
            if self.checkpoint_all:
                fts = torch.utils.checkpoint.checkpoint(layer, fts, use_reentrant=False)  # type: ignore
            else:
                fts = layer(fts)

            # UNet residuals logic (skip connections)
            if count < self.num_steps:
                if isinstance(layer, CoreBlock):
                    skip_inputs[count] = fts
                    count += 1
            elif count >= self.num_steps:
                if (
                    isinstance(layer, BilinearUpsample)
                    or isinstance(layer, TransposedConvUpsample)
                    or isinstance(layer, ZonallyPeriodicBilinearUpsample)
                ):
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
                    fts += skip_inputs[int(2 * self.num_steps - count - 1)]
                    count += 1

        return fts
