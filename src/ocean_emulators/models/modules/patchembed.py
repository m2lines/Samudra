# Sources inspired by the following implementations:
# - https://github.com/microsoft/aurora/blob/main/aurora/model/patchembed.py
# - https://github.com/lucidrains/vit-pytorch

from einops.layers.torch import Rearrange
from jaxtyping import Array, Float
from torch import nn

from ocean_emulators.constants import Input


class PatchEmbed2d(nn.Module):
    """A patch embedding for Samudra's flattened data (the channel dim is a cross of variable, level, and time)."""

    def __init__(
        self,
        input_vars: list[str],
        patch_size: int | tuple[int, int] = 4,
        embed_dim: int = 1024,
        hist: int = 1,
        norm: type[nn.Module] | None = nn.LayerNorm,
    ) -> None:
        """Embed TrainData arrays into patches.

        Args:
            input_vars (list[str]): list of input variable names. For input, this is typically the target prognostic
              and boundary variable names.
            patch_size (int): the size of the patches to embed. Patches must evenly divide the input grid. Further, the
              grid dimension divided by the patch size must be greater than 16 pixels.
            embed_dim (int): the dimension of the embedding to use.
            hist (int): for the input channels, the number of additional time steps to include. With `hist=0`, it will
              only include the present timestep. With `hist=1`, it will include the present and previous time step.
            norm (type[nn.Module]): the normalization layer to use. This is applied both after creating patches and
              after performing the linear projection.
        """
        super().__init__()
        if isinstance(patch_size, int):
            self.patch_size: tuple[int, int] = (patch_size, patch_size)
        else:
            assert isinstance(patch_size, tuple) and len(patch_size) == 2, (
                "Patch sizes must only span spatial dimensions (lat and lon)!"
            )
            self.patch_size = patch_size

        self.embed_dim: int = embed_dim
        self.n_channels = len(input_vars) * (1 + hist)

        self.n_patches = patch_dim = (
            self.n_channels * self.patch_size[0] * self.patch_size[1]
        )

        # While we could perform a patch embedding and linear projection in one step with a convolution, this
        # implementation is much clearer. I don't expect the additional computational cost to be arduous.
        self.patches = Rearrange(
            "b v (h ph) (w pw) -> b (h w) (ph pw v)",
            ph=self.patch_size[0],
            pw=self.patch_size[1],
        )
        self.norm_patches = norm(patch_dim) if norm is not None else nn.Identity()
        self.linear = nn.Linear(patch_dim, embed_dim)
        self.norm_embedding = norm(embed_dim) if norm is not None else nn.Identity()

    def forward(self, x: Input) -> Float[Array, "*batch patch_dim {self.embed_dim}"]:
        B, V, H, W = x.shape

        # V is a cross product of variable, level (encoded in vars), and time (has history).
        assert V == self.n_channels

        # Ensure patch_size is appropriate for the data.
        assert H % self.patch_size[0] == 0, f"{H} %  {self.patch_size[0]} != 0."
        assert (h := (H // self.patch_size[0]) ** 2) and h >= 16, (
            f"{h=}: A picture is work 16x16 words!"
        )
        assert W % self.patch_size[1] == 0, f"{W} %  {self.patch_size[1]} != 0."
        assert (w := (W // self.patch_size[1]) ** 2) and w >= 16, (
            f"{w=}: A picture is work 16x16 words!"
        )

        x = self.patches(x)
        x = self.norm_patches(x)
        x = self.linear(x)
        x = self.norm_embedding(x)

        return x
