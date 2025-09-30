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
        patch_size: tuple[int, int],
        perceiver_depth: int,
        perceiver_latent_dim: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size

        self.norm_patches = nn.LayerNorm(self.in_channels)
        self.perceiver = Perceiver(
            num_freq_bands=4,
            max_freq=self.in_channels,
            depth=perceiver_depth,
            input_axis=1,  # Number of positional dims before token dim
            input_channels=self.in_channels,
            latent_dim=perceiver_latent_dim,
            num_classes=out_channels * patch_size[0] * patch_size[1],
            weight_tie_layers=True,  # share weights of cross-attn blocks
            self_per_cross_attn=2,  # ratio of self-attn (latent, small) and cross-attn (input, big) blocks
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_patches_h_w = x.shape[2:]
        x = rearrange(
            x, "b l h w -> (b h w) l 1"
        )  # h/w here are in units of patches; each patch has l tokens with 1 element each (will get pos encoding internally)
        x = self.norm_patches(x)
        x = self.perceiver(x)
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
