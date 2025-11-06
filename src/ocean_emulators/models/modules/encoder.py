# Sources inspired by the following implementations:
# - https://github.com/microsoft/aurora/blob/main/aurora/model/patchembed.py
# - https://github.com/microsoft/aurora/blob/main/aurora/model/encoder.py
# - https://github.com/lucidrains/vit-pytorch

import torch
from aurora.model.fourier import pos_expansion, scale_expansion
from aurora.model.posencoding import pos_scale_enc
from einops import rearrange
from jaxtyping import Float
from torch import nn

from ocean_emulators.constants import Input, Lat, Lon


class PerceiverEncoder(nn.Module):
    """A perceiver-based encoder for Samudra's flattened data (a whole column of the ocean, with history).

    We adopt some of Aurora's positional encodings[1], which uses log-spaced fourier features with geometry-informed
    wavelengths. These encode 2d positions (the average latitude and longitude of each patch) as well as grid cell area
    (measured in km^2) for each token before it enters the processor.

    Args:
        in_channels (int): the number of input channels (roughly:  time x variable x (surface + depths)).
        out_channels (int): size of the latent dimension (aka, the embedding dimension).
        patch_size (int | tuple[int, int]): the size of the patches to embed. Patches must evenly divide the input grid.
          If a tuple is supplied, then it represents the (height, width) of the patches to embed.
        perceiver (nn.Module): the perceiver module implementation to use.

    References:
        [1]: https://ar5iv.labs.arxiv.org/html/2405.13063#A2.SS4
    """

    # TODO(alxmrs): Implement gradient checkpointing
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int | tuple[int, int],
        perceiver: nn.Module,
        lat: Lat,
        lon: Lon,
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
        self.lat, self.lon = lat, lon
        self.pos_embed = nn.Linear(self.out_channels, self.out_channels)
        self.scale_embed = nn.Linear(self.out_channels, self.out_channels)

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
        # NB(alxmrs): This is includes a mean and LayerNorm before linear projection!
        x = self.perceiver(x)  # (B_H_W, PH, PW, V) -> (B_H_W, out_channels)

        pos_encode, scale_encode = pos_scale_enc(
            self.out_channels,  # aka "embed_dim"
            self.lat,
            self.lon,
            self.patch_size,
            # TODO(#452): Pos and scale wavelengths range all the way to the whole Earth by default; we could probably
            #  better tune these for our Oceans modeling use case.
            pos_expansion=pos_expansion,
            scale_expansion=scale_expansion,
        )
        pos_encoding = self.pos_embed(pos_encode.to(dtype=x.dtype))
        scale_encoding = self.scale_embed(scale_encode.to(dtype=x.dtype))
        x = x + pos_encoding + scale_encoding

        x = rearrange(
            x,
            "(b h w) l -> b l h w",
            h=(H // self.patch_size[0]),
            w=(W // self.patch_size[1]),
        )

        return x
