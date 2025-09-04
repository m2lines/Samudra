# Sources inspired by the following implementations:
# - https://github.com/microsoft/aurora/blob/main/aurora/model/patchembed.py
# - https://github.com/lucidrains/vit-pytorch
import torch
from einops.layers.torch import Rearrange
from jaxtyping import Float
from torch import nn

from ocean_emulators.constants import Input


class PatchEmbed2d(nn.Module):
    """A patch embedding for Samudra's flattened data (the channel dim is a cross of variable, level, and time).

    Args:
        input_vars (list[str]): list of input variable names. For input, this is typically the target prognostic
          and boundary variable names.
        patch_size (int): the size of the patches to embed. Patches must evenly divide the input grid.
        embed_dim (int): size of the latent dimension.
        hist (int): for the input channels, the number of additional time steps to include. With `hist=0`, it will
          only include the present timestep. With `hist=1`, it will include the present and previous time step.
    """

    def __init__(
        self,
        n_channels: int,
        patch_size: int | tuple[int, int] = 4,
        embed_dim: int = 1024,
    ) -> None:
        super().__init__()
        if isinstance(patch_size, int):
            self.patch_size: tuple[int, int] = (patch_size, patch_size)
        else:
            assert isinstance(patch_size, tuple) and len(patch_size) == 2, (
                "Patch sizes must only span spatial dimensions (lat and lon)!"
            )
            self.patch_size = patch_size

        self.n_channels = n_channels
        self.embed_dim: int = embed_dim

        patch_dim = self.n_channels * self.patch_size[0] * self.patch_size[1]

        # While we could perform a patch embedding and linear projection in one step with a convolution, this
        # implementation is much clearer. I don't expect the additional computational cost to be arduous.
        self.patches = Rearrange(
            "b v (h ph) (w pw) -> b (h w) (ph pw v)",
            ph=self.patch_size[0],
            pw=self.patch_size[1],
        )
        self.norm_patches = nn.LayerNorm(patch_dim)
        self.linear = nn.Linear(patch_dim, embed_dim)
        self.norm_embedding = nn.LayerNorm(embed_dim)

    def forward(
        self, x: Input
    ) -> Float[torch.Tensor, "*batch {self.embed_dim} patch_dim"]:
        B, V, H, W = x.shape

        # V is a cross product of variable, level (encoded in vars), and time (has history).
        assert V == self.n_channels

        # Ensure patch_size is appropriate for the data.
        assert H % self.patch_size[0] == 0, f"{H} %  {self.patch_size[0]} != 0."
        assert W % self.patch_size[1] == 0, f"{W} %  {self.patch_size[1]} != 0."

        x = self.patches(x)
        x = self.norm_patches(x)
        x = self.linear(x)
        x = self.norm_embedding(x)  # (batch, patch_dim, embed_dim)
        x = x.transpose(1, 2)  # (batch, embed_dim, patch_dim)

        return x
