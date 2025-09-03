import torch

from ocean_emulators.config import EncoderConfig


class Encoder(torch.nn.Module):
    def __init__(self, config: EncoderConfig, **kwargs):
        super().__init__()
        self.patch_size = config.patch_size
