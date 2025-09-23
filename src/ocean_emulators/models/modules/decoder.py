import torch
from einops import rearrange
from perceiver_pytorch import Perceiver
from torch import nn


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
        grid_size: tuple[int, int],
        perceiver_depth: int,
        perceiver_latent_dim: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size

        grid_area = grid_size[0] * grid_size[1]

        self.norm_patches = nn.LayerNorm(self.in_channels)
        self.perceiver = Perceiver(
            num_freq_bands=4,
            max_freq=10.0,  # Depending on patch size and grid, consider values ranging from 3-10.
            depth=perceiver_depth,
            input_axis=2,  # Number of positional dims before token dim
            input_channels=self.in_channels,
            latent_dim=perceiver_latent_dim,
            num_classes=out_channels * grid_area,
            weight_tie_layers=True,  # share weights of cross-attn blocks
            self_per_cross_attn=2,  # ratio of self-attn (latent, small) and cross-attn (input, big) blocks
        )
        self.norm_embedding = nn.LayerNorm(out_channels * grid_area)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b l h w -> b h w l")
        x = self.norm_patches(x)
        x = self.perceiver(x)
        x = self.norm_embedding(x)
        x = rearrange(
            x,
            "b (h w c) -> b c h w",
            c=self.out_channels,
            h=self.grid_size[0],
            w=self.grid_size[1],
        )

        return x
