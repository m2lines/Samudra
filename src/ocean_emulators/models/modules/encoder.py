# Sources inspired by the following implementations:
# - https://github.com/microsoft/aurora/blob/main/aurora/model/patchembed.py
# - https://github.com/lucidrains/vit-pytorch


import torch
from einops import rearrange
from jaxtyping import Float
from torch import nn

from ocean_emulators.constants import Input


class PerceiverEncoder(nn.Module):
    """A perceiver-based encoder for Samudra's flattened data (a whole column of the ocean, with history).

    Args:
        in_channels (int): the number of input channels (roughly:  time x variable x (surface + depths)).
        out_channels (int): size of the latent dimension (aka, the embedding dimension).
        patch_size (int | tuple[int, int]): the size of the patches to embed. Patches must evenly divide the input grid.
          If a tuple is supplied, then it represents the (height, width) of the patches to embed.
        perceiver (nn.Module): the perceiver module implementation to use.
    """

    # TODO(alxmrs): Implement gradient checkpointing
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int | tuple[int, int],
        perceiver: nn.Module,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        if isinstance(patch_size, int):
            self.patch_size: tuple[int, int] = (patch_size, patch_size)
        else:
            assert isinstance(patch_size, tuple) and len(patch_size) == 2, (
                "Patch sizes must only span spatial dimensions (lat and lon)!"
            )
            self.patch_size = patch_size
        self.out_channels: int = out_channels  # aka, `embed_dim`.
        self.perceiver = perceiver
        self.project = nn.Linear(perceiver.latent_dim, out_channels)

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
        x = self.perceiver(x)  # (B_H_W, PH, PW, V) -> (B_H_W, num_latents, latent_dim)
        # TODO(alxmrs,jder): Why compute the mean? Is it better to directly project from the num_latents x latent_dim?
        x = x.mean(dim=1)  # (B_H_W, num_latents, latent_dim) -> (B_H_W, latent_dim)
        x = self.project(x)  # (B_H_W, latent_dim) -> (B_H_W, out_channels)
        x = rearrange(
            x,
            "(b h w) l -> b l h w",
            h=(H // self.patch_size[0]),
            w=(W // self.patch_size[1]),
        )

        return x
