# Sources inspired by the following implementations:
# - https://github.com/microsoft/aurora/blob/main/aurora/model/patchembed.py
# - https://github.com/lucidrains/vit-pytorch

import torch
from einops import rearrange
from jaxtyping import Float
from perceiver_pytorch import Perceiver
from torch import nn

from ocean_emulators.constants import Input


class PerceiverEncoder(nn.Module):
    """A perceiver-based encoder for Samudra's flattened data (a whole column of the ocean, with history).

    Args:
        in_channels (int): the number of input channels (roughly:  time x variable x (surface + depths)).
        out_channels (int): size of the latent dimension (aka, the embedding dimension).
        patch_size (int | tuple[int, int]): the size of the patches to embed. Patches must evenly divide the input grid.
          If a tuple is supplied, then it represents the (height, width) of the patches to embed.
        perceiver_depth (int): depth of the perceiver module core.
        perceiver_latent_dim (int): latent dimension of the perceiver module core. The `N` of the Perceiver's `O(M*N)`
          complexity, where the `M` corresponds to the size of the input data.
    """

    # TODO(alxmrs): Implement gradient checkpointing
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int | tuple[int, int],
        perceiver_depth: int,
        perceiver_latent_dim: int,
        perceiver_num_latents: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels: int = out_channels  # aka, `embed_dim`.

        if isinstance(patch_size, int):
            self.patch_size: tuple[int, int] = (patch_size, patch_size)
        else:
            assert isinstance(patch_size, tuple) and len(patch_size) == 2, (
                "Patch sizes must only span spatial dimensions (lat and lon)!"
            )
            self.patch_size = patch_size

        self.norm_patches = nn.LayerNorm(self.in_channels)
        self.perceiver = Perceiver(
            num_freq_bands=4,
            max_freq=max(
                *self.patch_size
            ),  # This is not actually a "frequency" but a maximum of the width appears to be reasonable from looking at the code
            depth=perceiver_depth,
            input_axis=2,  # Number of positional dims before token dim
            input_channels=self.in_channels,
            latent_dim=perceiver_latent_dim,
            num_latents=perceiver_num_latents,
            num_classes=out_channels,
            weight_tie_layers=False,  # share weights of cross-attn blocks
            self_per_cross_attn=2,  # ratio of self-attn (latent, small) and cross-attn (input, big) blocks
        )
        self.norm_embedding = nn.LayerNorm(out_channels)

    def forward(self, x: Input) -> Float[torch.Tensor, "batch {self.embed_dim} h w"]:
        _, V, H, W = x.shape

        # V is a cross product of variable, level (encoded in vars), and time (has history).
        assert V == self.in_channels
        # Ensure patch_size is appropriate for the data.
        assert H % self.patch_size[0] == 0, f"{H} % {self.patch_size[0]} != 0."
        assert W % self.patch_size[1] == 0, f"{W} % {self.patch_size[1]} != 0."

        # Perceiver experiment ideas:
        # 1. leave it as it is: treating each pixel as a token -- i.e. all channels (includes depths) per pixel
        # 2. change to original plan, where each float is its own token
        # 3. Add a third dim -- ph pw d v -- so each spatial position is a token
        x = rearrange(
            x,
            "b v (h ph) (w pw) -> (b h w) ph pw v",
            ph=self.patch_size[0],
            pw=self.patch_size[1],
        )
        # This is applying a layer norm across all our data channels. I am not sure if this will have a positive or
        # negative effect. It may make it easier for the Perceiver to process data across scales, but it destroys the
        # physical relationships in the data and aggregates across depth level and history. We should be cautious here.
        x = self.norm_patches(x)
        x = self.perceiver(x)
        x = self.norm_embedding(x)
        x = rearrange(
            x,
            "(b h w) l -> b l h w",
            h=(H // self.patch_size[0]),
            w=(W // self.patch_size[1]),
        )

        return x
