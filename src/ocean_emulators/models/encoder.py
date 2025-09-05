import torch
from torch import nn

from ocean_emulators.config import EncoderConfig
from ocean_emulators.constants import Input
from ocean_emulators.models.modules.patchembed import PerceiverPatchEmbed


class Encoder(torch.nn.Module):
    def __init__(self, config: EncoderConfig, n_channels: int, static_data):
        super().__init__()

        # TODO(alxmrs): Add static data to encoder!
        assert static_data is not None

        self.patch_embed = PerceiverPatchEmbed(
            n_channels,
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
            perceiver_depth=config.perceiver_depth,
        )
        self.dropout = nn.Dropout(p=config.positional_dropout_rate)

    # TODO(alxmrs): Implement position, scale, and time embeddings.
    def forward(self, x: Input) -> torch.Tensor:
        # The 'variable' dimension of `x` represents a whole column of the ocean: it's a cross product of time (history)
        # and data_var (a concatenation of boundary (surface) vars and prognostic vars, which itself is a cross product
        # of data_var and depth level).
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
