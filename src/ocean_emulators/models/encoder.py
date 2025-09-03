import torch

from ocean_emulators.config import EncoderConfig
from ocean_emulators.models.modules.patchembed import PatchEmbed2d


class Encoder(torch.nn.Module):
    def __init__(self, config: EncoderConfig, input_vars: list[str]):
        super().__init__()

        self.patch_embed = PatchEmbed2d(
            input_vars, patch_size=config.patch_size, embed_dim=config.patch_embed_dim
        )
