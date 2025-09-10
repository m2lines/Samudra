# Sources inspired by the following implementations:
# - https://github.com/microsoft/aurora/blob/main/aurora/model/patchembed.py
# - https://github.com/lucidrains/vit-pytorch
from typing import Any

import torch
from einops import rearrange
from jaxtyping import Float
from perceiver_pytorch import Perceiver
from torch import nn

from ocean_emulators.constants import Input


class PerceiverEncoder(nn.Module):
    """A perceiver-based encoder for Samudra's flattened data (a whole column of the ocean, with history).

    Args:
        n_channels (int): the number of input channels (roughly:  time x variable x (surface + depths)).
        patch_size (int): the size of the patches to embed. Patches must evenly divide the input grid.
        embed_dim (int): size of the latent dimension.
        perceiver_depth (int): depth of the perceiver module core.
    """

    def __init__(
        self,
        n_channels: int,
        patch_size: int | tuple[int, int],
        embed_dim: int,
        perceiver_depth: int,
        **perceiver_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        if isinstance(patch_size, int):
            self.patch_size: tuple[int, int] = (patch_size, patch_size)
        else:
            assert isinstance(patch_size, tuple) and len(patch_size) == 2, (
                "Patch sizes must only span spatial dimensions (lat and lon)!"
            )
            self.patch_size = patch_size
        self.embed_dim: int = embed_dim

        self.norm_patches = nn.LayerNorm(self.n_channels)
        self.perceiver = Perceiver(
            num_freq_bands=4,
            max_freq=1.0,
            depth=perceiver_depth,
            input_axis=2,  # Number of positional dims before token dim
            input_channels=self.n_channels,  # input_dim
            num_classes=embed_dim,  # output_dim
        )
        self.norm_embedding = nn.LayerNorm(embed_dim)

    def forward(self, x: Input) -> Float[torch.Tensor, "*batch h w {self.embed_dim}"]:
        _, V, H, W = x.shape

        # V is a cross product of variable, level (encoded in vars), and time (has history).
        assert V == self.n_channels
        # Ensure patch_size is appropriate for the data.
        assert H % self.patch_size[0] == 0, f"{H} %  {self.patch_size[0]} != 0."
        assert W % self.patch_size[1] == 0, f"{W} %  {self.patch_size[1]} != 0."

        # Perceiver experiment ideas:
        # 1. leave it as it is: treating each pixel as a token -- i.e. all channels (includes depths) per pixel
        # 2. change to original plan, where each float is its own token
        # 3. Add a third dim -- ph pw d v -- so each spatial position is a token
        x = rearrange(
            x,
            "b v (h ph) (w pw) -> (b h w) ph pw v",
            pw=self.patch_size[0],
            ph=self.patch_size[1],
        )
        # This is applying a layer norm across all our data channels. I am not sure if this will have a positive or
        # negative effect. It may make it easier for the Perceiver to process data across scales, but it destroys the
        # physical relationships in the data and aggregates across depth level and history. We should be cautious here.
        x = self.norm_patches(x)
        x = self.perceiver(x)
        x = rearrange(
            x,
            "(b h w) l -> b h w l",
            h=(H // self.patch_size[0]),
            w=(W // self.patch_size[1]),
        )
        x = self.norm_embedding(x)

        return x
