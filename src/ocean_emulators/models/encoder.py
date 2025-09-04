import torch
from torch import nn

from ocean_emulators.config import EncoderConfig
from ocean_emulators.constants import Input
from ocean_emulators.models.modules.patchembed import PatchEmbed2d


class Encoder(torch.nn.Module):
    def __init__(self, config: EncoderConfig, input_vars: list[str], static_data):
        super().__init__()

        assert static_data is not None, "TODO(alxmrs): Add static data to encoder!"
        self.patch_embed = PatchEmbed2d(
            input_vars, patch_size=config.patch_size, embed_dim=config.patch_embed_dim
        )

        self.dropout = nn.Dropout(p=config.positional_dropout_rate)

    # TODO(alxmrs): Implement position, scale, and time embeddings.
    def forward(self, x: Input) -> torch.Tensor:
        x = self.patch_embed(x)

        # Add position and scale embeddings to the patch embedding.
        pos_embed, scale_embed = torch.zeros_like(x), torch.zeros_like(x)
        x = x + pos_embed + scale_embed

        # Add lead time embeddings
        lead_time_embed = torch.zeros_like(x)
        x = x + lead_time_embed

        # Add absolute time embedding
        absolute_time_embed = torch.zeros_like(x)
        x = x + absolute_time_embed

        x = self.dropout(x)

        return x
