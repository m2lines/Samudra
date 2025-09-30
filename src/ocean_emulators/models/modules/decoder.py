from typing import final

import torch
from einops import rearrange
from perceiver_pytorch import PerceiverIO
from torch import nn


@final
class PerceiverDecoder(nn.Module):
    """A perceiver-based decoder that maps processed latent predictions into a whole column of the ocean.

    Args:
        in_channels (int): the number of input channels (typically, the output of our UNet backbone).
        out_channels (int): size of our output channels (roughly: variables x depths).
        grid_size (tuple[int, int]): size of the final output grid (lat / lng).
        perceiver_depth (int): depth of the perceiver module core.
        perceiver_latent_dim (int): latent dimension of the perceiver module core. The `N` of the Perceiver's `O(M*N)`
          complexity, where the `M` corresponds to the size of the input data.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: tuple[int, int],
        perceiver_depth: int,
        perceiver_latent_dim: int,
        perceiver_num_latents: int,
        output_channel_metadata: torch.Tensor,
        num_freq_bands: int = 6,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.num_freq_bands = num_freq_bands

        # Query dimension: (sin+cos for x, sin+cos for y) * num_freq_bands + depth + var_kind + time_delta
        self.queries_dim = 2 * num_freq_bands * 2 + 3

        self.norm_patches = nn.LayerNorm([self.in_channels, 1])
        self.perceiver = PerceiverIO(
            dim=self.in_channels,
            queries_dim=self.queries_dim,
            depth=perceiver_depth,
            latent_dim=perceiver_latent_dim,
            num_latents=perceiver_num_latents,
            weight_tie_layers=False,  # share weights of cross-attn blocks
        )

        # Register buffer for output channel metadata (depth, var_kind, time_delta)
        self.output_channel_metadata = torch.nn.Buffer(output_channel_metadata)

        # Register buffer for query positions (will be created on first forward pass)
        self.query_positions = torch.nn.Buffer(persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        num_patches_h_w = x.shape[2:]

        # Create queries with positional encodings: (c, p_h, p_w) for each output point
        # Queries shape: (c * p_h * p_w, queries_dim) where queries_dim = 3 (x_pos, y_pos, channel_idx)
        num_queries = self.out_channels * self.patch_size[0] * self.patch_size[1]

        if self.query_positions is None or self.query_positions.shape[0] != num_queries:
            # Create query positions with sinusoidal embeddings
            queries = torch.zeros(
                num_queries, self.queries_dim, device=x.device, dtype=x.dtype
            )

            idx = 0
            for c in range(self.out_channels):
                for p_h in range(self.patch_size[0]):
                    for p_w in range(self.patch_size[1]):
                        # Normalize positions to [0, 1]
                        pos_h = (
                            p_h / self.patch_size[0] if self.patch_size[0] > 1 else 0.5
                        )
                        pos_w = (
                            p_w / self.patch_size[1] if self.patch_size[1] > 1 else 0.5
                        )

                        # Create sinusoidal embeddings for x/y positions
                        for freq_idx in range(self.num_freq_bands):
                            freq = 2**freq_idx
                            # x position: sin and cos
                            queries[idx, freq_idx * 2] = torch.sin(
                                torch.tensor(pos_h * torch.pi * freq)
                            )
                            queries[idx, freq_idx * 2 + 1] = torch.cos(
                                torch.tensor(pos_h * torch.pi * freq)
                            )
                            # y position: sin and cos
                            queries[idx, self.num_freq_bands * 2 + freq_idx * 2] = (
                                torch.sin(torch.tensor(pos_w * torch.pi * freq))
                            )
                            queries[idx, self.num_freq_bands * 2 + freq_idx * 2 + 1] = (
                                torch.cos(torch.tensor(pos_w * torch.pi * freq))
                            )

                        # Add channel metadata (depth, var_kind, time_delta)
                        # Metadata: [depth, var_kind, time_delta]
                        depth = self.output_channel_metadata[c, 0]
                        var_kind = self.output_channel_metadata[c, 1]
                        time_delta = self.output_channel_metadata[c, 2]

                        # Normalize depth to roughly [-1, 1] (max depth ~6000m)
                        queries[idx, -3] = depth / 3000.0 - 1.0
                        # Normalize var_kind to [-1, 1] (0-4 range)
                        queries[idx, -2] = (var_kind / 2.0) - 1.0
                        # Normalize time_delta to [-1, 1] (typically -10 to 0 days)
                        queries[idx, -1] = time_delta / 10.0
                        idx += 1

            self.query_positions = queries

        # Prepare input: (b h w) l 1
        x = rearrange(x, "b l h w -> (b h w) l 1")

        # Expand queries for each batch*patch: (b h w, num_queries, queries_dim)
        queries = self.query_positions.unsqueeze(0).expand(x.shape[0], -1, -1)

        # Run PerceiverIO with queries
        x = self.perceiver(x, queries=queries)

        # Reshape output back to spatial layout
        x = rearrange(
            x,
            "(b h w) (c p_h p_w) -> b c (h p_h) (w p_w)",
            c=self.out_channels,
            p_h=self.patch_size[0],
            p_w=self.patch_size[1],
            h=num_patches_h_w[0],
            w=num_patches_h_w[1],
        )

        return x
