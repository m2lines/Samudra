import torch
from einops import rearrange
from perceiver_pytorch import Perceiver
from torch import nn


class PerceiverDecoder(nn.Module):
    """A perceiver-based decoder that maps processed latent predictions into a whole column of the ocean.

    Args:
        in_channels (int): the number of input channels (typically, the output of our UNet backbone).
        out_channels (int): size of our output channels (roughly: variables x depths).
        patch_size (int | tuple[int, int]): the size of the patches to embed. Patches must evenly divide the input grid.
          If a tuple is supplied, then it represents the (height, width) of the patches to embed.
        perceiver_depth (int): depth of the perceiver module core.
        perceiver_latent_dim (int): latent dimension of the perceiver module core. The `N` of the Perceiver's `O(M*N)`
          complexity, where the `M` corresponds to the size of the input data.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int | tuple[int, int],
        perceiver_depth: int,
        perceiver_latent_dim: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(patch_size, int):
            self.patch_size: tuple[int, int] = (patch_size, patch_size)
        else:
            assert isinstance(patch_size, tuple) and len(patch_size) == 2, (
                "Patch sizes must only span spatial dimensions (lat and lon)!"
            )
            self.patch_size = patch_size

        patch_area = self.patch_size[0] * self.patch_size[1]

        self.norm_patches = nn.LayerNorm(self.in_channels)
        self.perceiver = Perceiver(
            num_freq_bands=4,
            max_freq=10.0,  # Depending on patch size and grid, consider values ranging from 3-10.
            depth=perceiver_depth,
            input_axis=2,  # Number of positional dims before token dim
            input_channels=self.in_channels,
            latent_dim=perceiver_latent_dim,
            num_classes=out_channels * patch_area,
            weight_tie_layers=True,  # share weights of cross-attn blocks
            self_per_cross_attn=2,  # ratio of self-attn (latent, small) and cross-attn (input, big) blocks
        )
        self.norm_embedding = nn.LayerNorm(out_channels * patch_area)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, h, w = x.shape

        x = rearrange(x, "b l h w -> b h w l")
        x = self.norm_patches(x)
        x = self.perceiver(x)
        x = self.norm_embedding(x)

        # Reshape from flattened output back to spatial patches, then unpatchify
        # Perceiver outputs (B, out_channels * patch_area) for global pooling
        x = rearrange(
            x,
            "b (c ph pw) -> b c ph pw",
            c=self.out_channels,
            ph=self.patch_size[0],
            pw=self.patch_size[1],
        )

        # Repeat the patch content across the spatial grid
        x = x.repeat(1, 1, h, w)  # (B, C, ph, pw) -> (B, C, h*ph, w*pw)

        return x
