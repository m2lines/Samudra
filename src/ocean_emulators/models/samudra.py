import numpy as np
import torch
import torch.nn as nn

from ocean_emulators.models.base import BaseModel
from ocean_emulators.models.corrector import Corrector
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


class Samudra(BaseModel):
    def __init__(self, config, hist, wet, area_weights):
        super().__init__(
            ch_width=config.ch_width,
            n_out=config.n_out,
            wet=wet,
            hist=hist,
            pred_residuals=config.pred_residuals,
            last_kernel_size=config.last_kernel_size,
            pad=config.pad,
        )

        # Get activation class
        activation = get_activation_cl(config.core_block.activation)

        # Create local copies of config lists that will be reversed
        ch_width = config.ch_width.copy()
        dilation = config.dilation.copy()
        n_layers = config.n_layers.copy()

        # going down
        layers = []
        for i, (a, b) in enumerate(pairwise(ch_width)):
            # Core block
            layers.append(
                create_block(
                    config.core_block.block_type,
                    in_channels=a,
                    out_channels=b,
                    kernel_size=config.core_block.kernel_size,
                    dilation=dilation[i],
                    n_layers=n_layers[i],
                    activation=activation,
                    pad=config.pad,
                    upscale_factor=config.core_block.upscale_factor,
                    norm=config.core_block.norm,
                )
            )
            # Down sampling block
            layers.append(create_downsample(config.down_sampling_block))

        # Middle block
        layers.append(
            create_block(
                config.core_block.block_type,
                in_channels=b,
                out_channels=b,
                kernel_size=config.core_block.kernel_size,
                dilation=dilation[i],
                n_layers=n_layers[i],
                activation=activation,
                pad=config.pad,
                upscale_factor=config.core_block.upscale_factor,
                norm=config.core_block.norm,
            )
        )

        # First upsampling
        layers.append(
            create_upsample(config.up_sampling_block, in_channels=b, out_channels=b)
        )

        # Reverse for upsampling path
        ch_width.reverse()
        dilation.reverse()
        n_layers.reverse()

        # going up
        for i, (a, b) in enumerate(pairwise(ch_width[:-1])):
            layers.append(
                create_block(
                    config.core_block.block_type,
                    in_channels=a,
                    out_channels=b,
                    kernel_size=config.core_block.kernel_size,
                    dilation=dilation[i],
                    n_layers=n_layers[i],
                    activation=activation,
                    pad=config.pad,
                    upscale_factor=config.core_block.upscale_factor,
                    norm=config.core_block.norm,
                )
            )
            layers.append(
                create_upsample(config.up_sampling_block, in_channels=b, out_channels=b)
            )

        # Final conv block
        layers.append(
            create_block(
                config.core_block.block_type,
                in_channels=b,
                out_channels=b,
                kernel_size=config.core_block.kernel_size,
                dilation=dilation[i],
                n_layers=n_layers[i],
                activation=activation,
                pad=config.pad,
                upscale_factor=config.core_block.upscale_factor,
                norm=config.core_block.norm,
            )
        )

        # Final output conv
        layers.append(nn.Conv2d(b, config.n_out, config.last_kernel_size))

        self.layers = nn.ModuleList(layers)
        self.corrector = Corrector(config.corrector, hist, area_weights)
        self.num_steps = int(len(config.ch_width) - 1)

    def forward_once(self, fts):
        fts_input = fts.clone()
        temp: list[torch.Tensor] = []
        for i in range(self.num_steps):
            temp.append(torch.zeros_like(fts))
        count = 0
        for layer in self.layers:
            crop = fts.shape[2:]
            if isinstance(layer, nn.Conv2d):
                fts = torch.nn.functional.pad(
                    fts, (self.N_pad, self.N_pad, 0, 0), mode=self.pad
                )
                fts = torch.nn.functional.pad(
                    fts, (0, 0, self.N_pad, self.N_pad), mode="constant"
                )
            fts = layer(fts)
            if count < self.num_steps:
                if isinstance(layer, CoreBlock):
                    temp[count] = fts
                    count += 1
            elif count >= self.num_steps:
                if isinstance(layer, BilinearUpsample) or isinstance(
                    layer, TransposedConvUpsample
                ):
                    crop = np.array(fts.shape[2:])
                    shape = np.array(
                        temp[int(2 * self.num_steps - count - 1)].shape[2:]
                    )
                    pads = shape - crop
                    pads = [
                        pads[1] // 2,
                        pads[1] - pads[1] // 2,
                        pads[0] // 2,
                        pads[0] - pads[0] // 2,
                    ]
                    fts = nn.functional.pad(fts, pads)
                    fts += temp[int(2 * self.num_steps - count - 1)]
                    count += 1
        fts = self.corrector(fts_input, fts)
        return torch.where(self.wet, fts, 0.0)
