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

from ocean_emulators.constants import Boundary, Lat, Lon, Prognostic


def patch_from(
    patch_extent: tuple[float, float], input_height: int, input_width: int
) -> tuple[int, int]:
    """Calculate the patch size in lat/lng pixels (or coords) from the patch spatial extent and input grid size."""
    lat_spacing = 180.0 / input_height  # Full sphere is 180 degrees (pole to pole)
    lon_spacing = 360.0 / input_width  # Full circle is 360 degrees

    # Calculate patch size to match target extent
    patch_h = int(round(patch_extent[0] / lat_spacing))
    patch_w = int(round(patch_extent[1] / lon_spacing))

    return patch_h, patch_w


class PerceiverEncoder(nn.Module):
    """A dual-perceiver encoder that fuses prognostic and boundary streams.

    Each stream gets its own Perceiver.  The prognostic Perceiver is the
    primary workhorse; the boundary Perceiver is typically configured to be
    lightweight (fewer latents, shallower depth).  Both operate on 2-D
    spatial patches with ``input_axis=2``, so they get native Fourier
    position encoding within each patch for free.

    Because ``patch_extent`` is specified in degrees, both streams produce
    the **same latent grid** regardless of their spatial resolution.

    After mean-pooling each Perceiver's latents independently, the two
    representations are concatenated and projected back to ``latent_dim``
    via a learned linear layer.

    Patch-level positional and scale encodings (Aurora-style Fourier
    features [1]) are computed on the prognostic (output) grid and added
    after fusion.

    Args:
        prog_channels: Number of prognostic input channels.
        boundary_channels: Number of boundary input channels.
        out_channels: Size of the output embedding dimension.
        prog_latent_dim: Output dimension of the prognostic Perceiver.
        boundary_latent_dim: Output dimension of the boundary Perceiver.
        patch_extent: Spatial extent of each patch in degrees ``(lat, lon)``.
        perceiver: Perceiver module for the prognostic stream.
        boundary_perceiver: Perceiver module for the boundary stream.

    References:
        [1]: https://ar5iv.labs.arxiv.org/html/2405.13063#A2.SS4
    """

    # TODO(alxmrs): Implement gradient checkpointing
    def __init__(
        self,
        prog_channels: int,
        boundary_channels: int,
        out_channels: int,
        prog_latent_dim: int,
        boundary_latent_dim: int,
        patch_extent: tuple[float, float],
        perceiver: nn.Module,
        boundary_perceiver: nn.Module,
    ) -> None:
        super().__init__()
        self.prog_channels = prog_channels
        self.boundary_channels = boundary_channels
        self.out_channels: int = out_channels
        self.patch_extent = patch_extent
        self.perceiver = perceiver
        self.boundary_perceiver = boundary_perceiver

        # Fuse the two Perceiver outputs and project to embedding dim.
        self.fusion_proj = nn.Linear(
            prog_latent_dim + boundary_latent_dim, out_channels
        )

        # TODO(#451): The input to these position and scale linear units could be a hparam.
        self.pos_embed = nn.Linear(out_channels, out_channels)
        self.scale_embed = nn.Linear(out_channels, out_channels)

    def _patchify_params(
        self, shape: torch.Size, expected_channels: int
    ) -> tuple[int, int, int, int]:
        """Validate channels and compute patch / latent-grid dims.

        Returns ``(ph, pw, lat_h, lat_w)``.
        """
        _, v, h, w = shape
        assert v == expected_channels, (
            f"Expected {expected_channels} channels, got {v}."
        )
        ph, pw = patch_from(self.patch_extent, h, w)
        assert h % ph == 0, f"{h} % {ph} != 0."
        assert w % pw == 0, f"{w} % {pw} != 0."
        return ph, pw, h // ph, w // pw

    def forward(
        self,
        prog: Prognostic,
        boundary: Boundary,
        prog_res: tuple[Lat, Lon],
    ) -> Float[torch.Tensor, "batch {self.out_channels} h w"]:
        # --- Prognostic stream: 2-D patches → Perceiver ---
        patch_h, patch_w, lat_h, lat_w = self._patchify_params(
            prog.shape, self.prog_channels
        )

        prog_patches = rearrange(
            prog,
            "b v (h ph) (w pw) -> (b h w) ph pw v",
            ph=patch_h,
            pw=patch_w,
        )
        prog_pooled = self.perceiver(prog_patches)  # (B_HW, latent_dim)

        # --- Boundary stream: 2-D patches → boundary Perceiver ---
        b_patch_h, b_patch_w, b_lat_h, b_lat_w = self._patchify_params(
            boundary.shape, self.boundary_channels
        )

        assert lat_h == b_lat_h and lat_w == b_lat_w, (
            f"Latent grid mismatch: prog ({lat_h}, {lat_w}) vs "
            f"boundary ({b_lat_h}, {b_lat_w}). Check that patch_extent "
            f"divides both grids evenly."
        )

        boundary_patches = rearrange(
            boundary,
            "b v (h ph) (w pw) -> (b h w) ph pw v",
            ph=b_patch_h,
            pw=b_patch_w,
        )
        boundary_pooled = self.boundary_perceiver(
            boundary_patches
        )  # (B_HW, latent_dim)

        # --- Fusion: concat pooled representations, project ---
        x = self.fusion_proj(
            torch.cat([prog_pooled, boundary_pooled], dim=-1)
        )  # (B_HW, out_channels)

        # --- Patch-level positional + scale encoding ---
        x = rearrange(x, "(b h w) l -> b (h w) l", h=lat_h, w=lat_w)
        lat, lon = prog_res
        pos_encode, scale_encode = pos_scale_enc(
            self.out_channels,
            lat,
            lon,
            (patch_h, patch_w),
            # TODO(#452): Pos and scale wavelengths range all the way to the whole Earth by default; we could probably
            #  better tune these for our Oceans modeling use case.
            pos_expansion=pos_expansion,
            scale_expansion=scale_expansion,
        )
        pos_encoding = self.pos_embed(
            pos_encode.to(dtype=x.dtype, device=x.device)
        ).unsqueeze(0)
        scale_encoding = self.scale_embed(
            scale_encode.to(dtype=x.dtype, device=x.device)
        ).unsqueeze(0)
        x = x + pos_encoding + scale_encoding

        # Unpack spatial dims, move channel dim to standard location.
        x = rearrange(x, "b (h w) l -> b l h w", h=lat_h, w=lat_w)

        return x
