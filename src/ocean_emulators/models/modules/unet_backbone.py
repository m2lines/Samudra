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
from ocean_emulators.models.modules.padding import apply_spatial_pad
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
        domain_parallel: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = ch_width[0]

        # Create local copies of config lists that will be reversed
        ch_width = [in_channels] + ch_width.copy()
        dilation = dilation.copy()
        n_layers = n_layers.copy()
        self.pad = pad
        self.domain_parallel = domain_parallel
        self.disable_middle_block = False
        self.unet_skip_mode = "keep"
        self.unet_skip_indices: set[int] = set()

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
                    domain_parallel=domain_parallel,
                    checkpoint_simple=checkpoint_simple,
                )
            )
            # Down sampling block
            layers.append(downsampling_block)

        # Middle block
        self.middle_layer_index = len(layers)
        layers.append(
            create_block(
                in_channels=b,
                out_channels=b,
                dilation=dilation[i],
                n_layers=n_layers[i],
                pad=pad,
                domain_parallel=domain_parallel,
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
                    domain_parallel=domain_parallel,
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
                domain_parallel=domain_parallel,
                checkpoint_simple=checkpoint_simple,
            )
        )

        first_block = layers[0]
        assert isinstance(first_block, CoreBlock)
        self.N_pad = first_block.N_pad
        self.layers = nn.ModuleList(layers)
        self.num_steps = int(len(ch_width) - 1)

    def configure_ablation(
        self,
        disable_middle_block: bool = False,
        skip_mode: str = "keep",
        skip_indices: set[int] | None = None,
    ) -> None:
        valid_skip_modes = {"keep", "drop_all", "drop_indices", "keep_indices"}
        if skip_mode not in valid_skip_modes:
            raise ValueError(
                f"Unknown UNet skip ablation mode `{skip_mode}`. "
                f"Expected one of {sorted(valid_skip_modes)}."
            )

        skip_indices = skip_indices or set()
        invalid_indices = sorted(
            index for index in skip_indices if index < 0 or index >= self.num_steps
        )
        if invalid_indices:
            raise ValueError(
                "UNet skip ablation indices must be between 0 and "
                f"{self.num_steps - 1}; got {invalid_indices}."
            )

        self.disable_middle_block = disable_middle_block
        self.unet_skip_mode = skip_mode
        self.unet_skip_indices = skip_indices

    def _uses_skip_connection(self, skip_index: int) -> bool:
        match self.unet_skip_mode:
            case "keep":
                return True
            case "drop_all":
                return False
            case "drop_indices":
                return skip_index not in self.unet_skip_indices
            case "keep_indices":
                return skip_index in self.unet_skip_indices
            case _:
                raise RuntimeError(f"Unhandled UNet skip mode {self.unet_skip_mode}")

    def forward(self, fts: torch.Tensor) -> torch.Tensor:
        skip_inputs: list[torch.Tensor] = []
        for i in range(self.num_steps):
            skip_inputs.append(torch.zeros_like(fts))
        count = 0
        for layer_index, layer in enumerate(self.layers):
            # Circular/Globe padding
            if isinstance(layer, nn.Conv2d):
                fts = apply_spatial_pad(fts, self.N_pad, self.pad)

            # (Maybe) apply checkpointing
            if self.disable_middle_block and layer_index == self.middle_layer_index:
                pass
            elif self.checkpoint_all:
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
                    skip_index = int(2 * self.num_steps - count - 1)
                    crop = np.array(fts.shape[2:])
                    shape = np.array(skip_inputs[skip_index].shape[2:])
                    pads = shape - crop
                    pads = [
                        pads[1] // 2,
                        pads[1] - pads[1] // 2,
                        pads[0] // 2,
                        pads[0] - pads[0] // 2,
                    ]
                    if any(pads):
                        fts = nn.functional.pad(fts, pads)
                    if self._uses_skip_connection(skip_index):
                        fts += skip_inputs[skip_index]
                    count += 1

        return fts
