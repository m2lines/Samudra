from typing import assert_never

import numpy as np
import torch
from torch import nn

from ocean_emulators.config import BlockConfig, Checkpointing
from ocean_emulators.models.modules.blocks import (
    BilinearUpsample,
    CoreBlock,
    TransposedConvUpsample,
)
from ocean_emulators.models.modules.factory import (
    create_block,
    create_downsample,
    create_upsample,
    get_activation_cl,
)
from ocean_emulators.utils.train import pairwise


class UNetBackbone(nn.Module):
    def __init__(
        self,
        ch_width: list[int],
        dilation: list[int],
        n_layers: list[int],
        pad: str,
        core_block: BlockConfig,
        down_sampling_block: str,
        up_sampling_block: str,
        checkpointing: Checkpointing | None,
    ):
        super().__init__()

        # Get activation class
        activation = get_activation_cl(core_block.activation)

        # Create local copies of config lists that will be reversed
        ch_width = ch_width.copy()
        dilation = dilation.copy()
        n_layers = n_layers.copy()

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
        layers = []
        for i, (a, b) in enumerate(pairwise(ch_width)):
            # Core block
            layers.append(
                create_block(
                    core_block.block_type,
                    in_channels=a,
                    out_channels=b,
                    kernel_size=core_block.kernel_size,
                    dilation=dilation[i],
                    n_layers=n_layers[i],
                    activation=activation,
                    pad=pad,
                    upscale_factor=core_block.upscale_factor,
                    norm=core_block.norm,
                    checkpoint_simple=checkpoint_simple,
                )
            )
            # Down sampling block
            layers.append(create_downsample(down_sampling_block))

        # Middle block
        layers.append(
            create_block(
                core_block.block_type,
                in_channels=b,
                out_channels=b,
                kernel_size=core_block.kernel_size,
                dilation=dilation[i],
                n_layers=n_layers[i],
                activation=activation,
                pad=pad,
                upscale_factor=core_block.upscale_factor,
                norm=core_block.norm,
                checkpoint_simple=checkpoint_simple,
            )
        )

        # First upsampling
        layers.append(create_upsample(up_sampling_block, in_channels=b, out_channels=b))

        # Reverse for upsampling path
        ch_width.reverse()
        dilation.reverse()
        n_layers.reverse()

        # going up
        for i, (a, b) in enumerate(pairwise(ch_width[:-1])):
            layers.append(
                create_block(
                    core_block.block_type,
                    in_channels=a,
                    out_channels=b,
                    kernel_size=core_block.kernel_size,
                    dilation=dilation[i],
                    n_layers=n_layers[i],
                    activation=activation,
                    pad=pad,
                    upscale_factor=core_block.upscale_factor,
                    norm=core_block.norm,
                    checkpoint_simple=checkpoint_simple,
                )
            )
            layers.append(
                create_upsample(up_sampling_block, in_channels=b, out_channels=b)
            )

        # Final conv block
        layers.append(
            create_block(
                core_block.block_type,
                in_channels=b,
                out_channels=b,
                kernel_size=core_block.kernel_size,
                dilation=dilation[i],
                n_layers=n_layers[i],
                activation=activation,
                pad=pad,
                upscale_factor=core_block.upscale_factor,
                norm=core_block.norm,
                checkpoint_simple=checkpoint_simple,
            )
        )

        self.layers = nn.ModuleList(layers)
        self.num_steps = int(len(ch_width) - 1)

    def forward(self, fts: torch.Tensor) -> torch.Tensor:
        skip_inputs: list[torch.Tensor] = []
        for i in range(self.num_steps):
            skip_inputs.append(torch.zeros_like(fts))
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
                if isinstance(layer, BilinearUpsample) or isinstance(
                    layer, TransposedConvUpsample
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
