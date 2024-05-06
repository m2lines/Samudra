import torch
from typing import Sequence
from hydra.utils import instantiate
from omegaconf import DictConfig
import numpy as np


class UNetDecoder(torch.nn.Module):
    """
    Generic UNetDecoder that can be applied to arbitrary meshes.
    """

    def __init__(
        self,
        conv_block: DictConfig,
        up_sampling_block: DictConfig,
        output_layer: DictConfig,
        n_channels: Sequence = (34, 68, 136),
        n_layers: Sequence = (1, 2, 2),
        output_channels: int = 3,
        dilations: list = (4, 2, 1),
    ):
        super().__init__()
        self.channel_dim = 1  # 1 in previous layout

        if dilations is None:
            # Defaults to [1, 1, 1...] in accordance with the number of unet levels
            dilations = [1 for _ in range(len(n_channels))]

        self.decoder = []
        for n, curr_channel in enumerate(n_channels):

            # Second half of the synoptic layer does not need an upsampling module
            if n == 0:
                up_sample_module = None
            else:
                up_sample_module = instantiate(
                    up_sampling_block,
                    in_channels=curr_channel,
                    out_channels=curr_channel,
                )
                # up_sample_module = torch.nn.Upsample(scale_factor=2, mode='bilinear')

            next_channel = (
                n_channels[n + 1] if n < len(n_channels) - 1 else n_channels[-1]
            )

            conv_module = instantiate(
                conv_block,
                in_channels=(
                    curr_channel * 2 if n > 0 else curr_channel
                ),  # Considering skip connection
                latent_channels=curr_channel,
                out_channels=next_channel,
                dilation=dilations[n],
                n_layers=n_layers[n],
            )

            self.decoder.append(
                torch.nn.ModuleDict(
                    {
                        "upsamp": up_sample_module,
                        "conv": conv_module
                    }
                )
            )

        self.decoder = torch.nn.ModuleList(self.decoder)

        # (Linear) Output layer
        self.output_layer = instantiate(
            output_layer,
            in_channels=curr_channel,
            out_channels=output_channels,
            dilation=dilations[-1],
            activation=None,
        )

    def forward(self, inputs: Sequence) -> torch.Tensor:
        x = inputs[-1]
        for n, layer in enumerate(self.decoder):
            if layer["upsamp"] is not None:
                up = layer["upsamp"](x)
                crop = np.array(up.shape[2:])
                shape = np.array(inputs[-1 - n].shape[2:])
                pads = (shape - crop)
                pads = [pads[1]//2, pads[1]-pads[1]//2,
                        pads[0]//2, pads[0]-pads[0]//2]
                up = torch.nn.functional.pad(up,pads)
                x =  torch.cat([up, inputs[-1 - n]], dim=self.channel_dim)
            x = layer["conv"](x)
        return self.output_layer(x)
