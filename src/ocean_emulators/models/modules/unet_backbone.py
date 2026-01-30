from typing import TYPE_CHECKING, assert_never

import numpy as np
import torch
from torch import nn

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
    from ocean_emulators.models.modules.token_conditioning import (  # noqa: F401
        DecoderFiLM,
        TokenConditioner,
    )


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
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = ch_width[0]

        ch_width_full = [in_channels] + ch_width.copy()
        self.bottleneck_channels = ch_width_full[-1]
        self.decoder_film_channels = list(reversed(ch_width_full[1:]))

        # Create local copies of config lists that will be reversed
        ch_width = ch_width_full
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

    def forward(self, fts: torch.Tensor) -> torch.Tensor:
        fts, _ = self._forward(fts, None, None, return_tokens=False)
        return fts

    def forward_with_tokens(
        self,
        fts: torch.Tensor,
        token_conditioner: "TokenConditioner",
        decoder_film: "DecoderFiLM | None",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._forward(fts, token_conditioner, decoder_film, return_tokens=True)

    def _forward(
        self,
        fts: torch.Tensor,
        token_conditioner: "TokenConditioner | None",
        decoder_film: "DecoderFiLM | None",
        return_tokens: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        skip_inputs: list[torch.Tensor] = []
        for i in range(self.num_steps):
            skip_inputs.append(torch.zeros_like(fts))
        count = 0
        decoder_block_idx = 0
        tokens: torch.Tensor | None = None
        tokens_concat: torch.Tensor | None = None
        for layer in self.layers:
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

            if (
                token_conditioner is not None
                and tokens is None
                and count == self.num_steps
                and isinstance(layer, CoreBlock)
            ):
                tokens = token_conditioner(fts)
                tokens_concat = tokens.reshape(tokens.shape[0], -1)

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
                    if decoder_film is not None:
                        if tokens_concat is None:
                            raise RuntimeError(
                                "decoder_film requires token_conditioner"
                            )
                        fts = decoder_film(decoder_block_idx, fts, tokens_concat)
                    decoder_block_idx += 1
                    count += 1

        if return_tokens:
            return fts, tokens
        return fts, None
