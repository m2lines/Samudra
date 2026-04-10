# Sources inspired by the following implementations:
# - https://github.com/microsoft/aurora/blob/main/aurora/model/patchembed.py
# - https://github.com/microsoft/aurora/blob/main/aurora/model/encoder.py
# - https://github.com/lucidrains/vit-pytorch


import torch
import torch.nn.functional as F
from aurora.model.fourier import pos_expansion, scale_expansion
from aurora.model.posencoding import pos_scale_enc
from einops import rearrange
from jaxtyping import Float
from perceiver_pytorch.perceiver_pytorch import FeedForward, PreNorm
from torch import nn

from ocean_emulators.constants import Boundary, Lat, Lon, Prognostic
from ocean_emulators.models.modules.augment_input import FourierPatchPositionEncoding


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


class MultiHeadAttention(nn.Module):
    """Multi-head attention using ``F.scaled_dot_product_attention``.

    Supports both self-attention (no context) and cross-attention
    (context provided as keys/values).
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim or query_dim

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)
        self.dropout = dropout

    def forward(
        self,
        x: Float[torch.Tensor, "batch seq query_dim"],
        context: Float[torch.Tensor, "batch ctx_seq context_dim"] | None = None,
    ) -> Float[torch.Tensor, "batch seq query_dim"]:
        context = context if context is not None else x
        h = self.heads

        q = rearrange(self.to_q(x), "b n (h d) -> b h n d", h=h)
        kv = self.to_kv(context)
        k, v = rearrange(kv, "b n (two h d) -> two b h n d", two=2, h=h)

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class PerceiverEncoder(nn.Module):
    """A perceiver-based encoder that fuses prognostic and boundary streams.

    The prognostic stream is the primary input.  It is patchified into 2-D
    spatial patches and fed directly to a Perceiver with ``input_axis=2``,
    preserving native 2-D Fourier position encoding within each patch.  The
    Perceiver's cross-attention layers do the heavy representation learning
    on raw prognostic channel data — no linear bottleneck before attention.

    The boundary stream is auxiliary context at a potentially different
    resolution.  It is patchified, linearly projected to the Perceiver's
    latent dimension, and injected via a **cross-attention layer** that lets
    the Perceiver's latent vectors attend to boundary tokens.  This
    asymmetric design reflects the asymmetric role of prognostics (high-res
    output to predict) vs. boundaries (low-res external forcing).

    Because ``patch_extent`` is specified in degrees, both streams produce
    the **same latent grid** regardless of their spatial resolution.

    Patch-level positional and scale encodings (Aurora-style Fourier
    features [1]) are computed on the prognostic (output) grid and added
    after the Perceiver and boundary fusion.

    Args:
        prog_channels: Number of prognostic input channels.
        boundary_channels: Number of boundary input channels.
        out_channels: Size of the output embedding dimension.
        latent_dim: Dimension of the Perceiver's internal latent vectors.
        patch_extent: Spatial extent of each patch in degrees ``(lat, lon)``.
        perceiver: Perceiver module operating on 2-D prog patches.  Must
            support ``perceiver(data, return_embeddings=True)`` returning
            ``(batch, num_latents, latent_dim)``.

    References:
        [1]: https://ar5iv.labs.arxiv.org/html/2405.13063#A2.SS4
    """

    # TODO(alxmrs): Implement gradient checkpointing
    def __init__(
        self,
        prog_channels: int,
        boundary_channels: int,
        out_channels: int,
        latent_dim: int,
        patch_extent: tuple[float, float],
        perceiver: nn.Module,
        boundary_attn_heads: int,
        num_fusion_self_attn: int,
        boundary_fourier_dim: int,
    ) -> None:
        super().__init__()
        self.prog_channels = prog_channels
        self.boundary_channels = boundary_channels
        self.in_channels: int = prog_channels + boundary_channels
        self.out_channels: int = out_channels
        self.latent_dim = latent_dim
        self.patch_extent = patch_extent
        self.perceiver = perceiver

        assert latent_dim % boundary_attn_heads == 0, (
            f"latent_dim ({latent_dim}) must be divisible by "
            f"boundary_attn_heads ({boundary_attn_heads})."
        )
        dim_head = latent_dim // boundary_attn_heads

        # Within each patch, boundary pixels get 2-D Fourier position
        # features so the cross-attention knows *where* in the patch each
        # token is.  The module adds 2 * fourier_dim features.
        self.boundary_pos_enc = FourierPatchPositionEncoding(boundary_fourier_dim)
        self.boundary_proj = nn.Linear(
            boundary_channels + 2 * boundary_fourier_dim, latent_dim
        )

        # Boundary: cross-attend from Perceiver latents.  PreNorm mirrors
        # the Perceiver's own layer structure.
        self.boundary_cross_attn = PreNorm(
            latent_dim,
            MultiHeadAttention(
                latent_dim,
                context_dim=latent_dim,
                heads=boundary_attn_heads,
                dim_head=dim_head,
            ),
            context_dim=latent_dim,
        )
        self.boundary_ff = PreNorm(latent_dim, FeedForward(latent_dim))

        # --- optional self-attention after fusion ---
        self.fusion_self_attn_layers = nn.ModuleList()
        for _ in range(num_fusion_self_attn):
            self.fusion_self_attn_layers.append(
                PreNorm(
                    latent_dim,
                    MultiHeadAttention(
                        latent_dim,
                        heads=boundary_attn_heads,
                        dim_head=dim_head,
                    ),
                )
            )
            self.fusion_self_attn_layers.append(
                PreNorm(latent_dim, FeedForward(latent_dim))
            )

        # Pool latents and project to output embedding dimension.
        self.latent_norm = nn.LayerNorm(latent_dim)
        self.to_out = nn.Linear(latent_dim, out_channels)

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
        ph_p, pw_p, lat_h, lat_w = self._patchify_params(prog.shape, self.prog_channels)

        prog_patches = rearrange(
            prog,
            "b v (h ph) (w pw) -> (b h w) ph pw v",
            ph=ph_p,
            pw=pw_p,
        )
        # NB(alxmrs): The Perceiver includes a mean and LayerNorm before
        # its linear projection, plus 2-D Fourier position encoding within
        # each patch.  return_embeddings=True gives us the raw latent
        # vectors *before* the Perceiver's output head.
        latents = self.perceiver(
            prog_patches, return_embeddings=True
        )  # (B_HW, num_latents, latent_dim)

        # --- Boundary stream: flatten patches, project ---
        ph_b, pw_b, b_lat_h, b_lat_w = self._patchify_params(
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
            ph=ph_b,
            pw=pw_b,
        )

        boundary_patches = self.boundary_pos_enc(boundary_patches)
        boundary_patches = rearrange(boundary_patches, "b ph pw d -> b (ph pw) d")
        boundary_tokens = self.boundary_proj(
            boundary_patches
        )  # (B_HW, ph_b*pw_b, latent_dim)

        # --- Fusion: Perceiver latents cross-attend to boundary context ---
        latents = self.boundary_cross_attn(latents, context=boundary_tokens) + latents
        latents = self.boundary_ff(latents) + latents

        # Optional self-attention layers for deeper fusion integration (turned off by default)
        for i in range(0, len(self.fusion_self_attn_layers), 2):
            latents = self.fusion_self_attn_layers[i](latents) + latents
            latents = self.fusion_self_attn_layers[i + 1](latents) + latents

        # --- Pool across latents and project to output dim ---
        x = latents.mean(dim=1)  # (B_HW, latent_dim)
        x = self.to_out(self.latent_norm(x))  # (B_HW, out_channels)

        # --- Patch-level positional + scale encoding ---
        x = rearrange(x, "(b h w) l -> b (h w) l", h=lat_h, w=lat_w)
        lat, lon = prog_res
        patch_h, patch_w = patch_from(self.patch_extent, len(lat), len(lon))
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
