import torch
from typing import Sequence
from hydra.utils import instantiate
from omegaconf import DictConfig

class UNetEncoder(torch.nn.Module):
    """
    Generic UNet3Encoder that can be applied to arbitrary meshes.
    """

    def __init__(
        self,
        conv_block: DictConfig,
        down_sampling_block: DictConfig,
        recurrent_block: DictConfig = None,
        input_channels: int = 3,
        n_channels: Sequence = (136, 68, 34),
        n_layers: Sequence = (2, 2, 1),
        dilations: list = (1, 2, 4),
    ):
        super().__init__()
        self.n_channels = n_channels

        if dilations is None:
            # Defaults to [1, 1, 1...] in accordance with the number of unet levels
            dilations = [1 for _ in range(len(n_channels))]

        # Build encoder
        old_channels = input_channels
        self.encoder = []
        for n, curr_channel in enumerate(n_channels):
            modules = list()
            if n > 0:
                modules.append(instantiate(down_sampling_block))
            else:
                down_pool_module = None

            modules.append(
                instantiate(
                    conv_block,
                    in_channels=old_channels,
                    latent_channels=curr_channel,
                    out_channels=curr_channel,
                    dilation=dilations[n],
                    n_layers=n_layers[n],
                )
            )

            old_channels = curr_channel

            self.encoder.append(torch.nn.Sequential(*modules))

        self.encoder = torch.nn.ModuleList(self.encoder)

    def forward(self, inputs: Sequence) -> Sequence:
        outputs = []
        for layer in self.encoder:
            outputs.append(layer(inputs))
            inputs = outputs[-1]
        return outputs

    def reset(self):
        pass
