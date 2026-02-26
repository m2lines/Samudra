# Mirrors the encoder structure in encoder.py
# Sources:
# - https://ar5iv.labs.arxiv.org/html/2405.13063 (Aurora paper, Appendix B.3: 3D Perceiver Decoder)
# - https://github.com/lucidrains/perceiver-pytorch

import torch
from aurora.model.fourier import pos_expansion, scale_expansion
from aurora.model.posencoding import pos_scale_enc
from einops import rearrange, repeat
from jaxtyping import Float
from torch import nn

from ocean_emulators.constants import Lat, Lon
from ocean_emulators.models.modules.encoder import patch_from


class PerceiverDecoder(nn.Module):
    """A perceiver-based decoder that maps latent patch tokens to full-resolution output.

    For each patch position on the latent grid, the decoder broadcasts the
    pos/scale-encoded latent vector to every output pixel within the patch,
    concatenates a normalized 2D pixel-position query, and runs a shared
    perceiver over the ``(patch_h, patch_w)`` grid of ``(C + 2)``-dim tokens.

    The perceiver's learned latents cross-attend to these pixel tokens and
    the output is mean-pooled to ``out_channels`` — producing one prediction
    per patch.  This is then reassembled into ``(B, out_channels, nh, nw)``.
    FOMO's ``unpatch`` linear handles the final expansion to full resolution.

    Because pixel queries are normalized to ``[0, 1)``, the same perceiver
    generalizes across resolutions: a 1-degree patch queries a coarse grid
    while a 0.25-degree patch queries a finer grid over the same coordinate
    space.

    Args:
        in_channels: Number of input channels from the processor.
        out_channels: Number of output channels per patch position.
        patch_extent: Spatial extent of each patch in degrees (lat, lon).
            Used for computing positional and scale encodings.
        perceiver: Shared perceiver module.  ``input_channels`` must equal
            ``in_channels + 2`` (latent dim + 2D pixel query).
            ``num_classes`` must equal ``out_channels``.

    References:
        [1]: https://ar5iv.labs.arxiv.org/html/2405.13063#A2.SS4
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_extent: tuple[float, float],
        perceiver: nn.Module,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_extent = patch_extent

        # Positional and scale encoding (mirrors encoder's post-perceiver encoding)
        self.pos_embed = nn.Linear(in_channels, in_channels)
        self.scale_embed = nn.Linear(in_channels, in_channels)

        self.perceiver = perceiver

    def forward(
        self,
        x: Float[torch.Tensor, "batch channels nh nw"],
        resolution: tuple[Lat, Lon],
    ) -> Float[torch.Tensor, "batch {self.out_channels} nh nw"]:
        # nh, nw: number of patches along height and width (the latent grid dims).
        # Not to be confused with patch_h, patch_w which are each patch's pixel size.
        B, C, nh, nw = x.shape
        lat, lon = resolution

        H, W = len(lat), len(lon)
        patch_h, patch_w = patch_from(self.patch_extent, H, W)

        # For pos/scale encoding we need a patch size that produces exactly
        # nh*nw tokens.  In the full pipeline patch_from gives the same result,
        # but in unit tests the input grid may not match, so derive from tensor.
        pos_patch_h, pos_patch_w = H // nh, W // nw

        # --- Add pos/scale encoding to latent tokens (mirrors encoder) ---
        tokens = rearrange(x, "b c nh nw -> b (nh nw) c")

        pos_encode, scale_encode = pos_scale_enc(
            C,
            lat,
            lon,
            (pos_patch_h, pos_patch_w),
            pos_expansion=pos_expansion,
            scale_expansion=scale_expansion,
        )
        pos_encoding = self.pos_embed(
            pos_encode.to(dtype=tokens.dtype, device=tokens.device)
        ).unsqueeze(0)
        scale_encoding = self.scale_embed(
            scale_encode.to(dtype=tokens.dtype, device=tokens.device)
        ).unsqueeze(0)
        tokens = tokens + pos_encoding + scale_encoding

        # --- Build per-pixel input: broadcast latent + concat pixel query ---
        # Broadcast each latent vector to every pixel in its patch.
        # tokens: (B, nh*nw, C) -> (B*nh*nw, patch_h, patch_w, C)
        tokens = rearrange(tokens, "b (nh nw) c -> (b nh nw) c", nh=nh, nw=nw)
        tokens = repeat(tokens, "n c -> n ph pw c", ph=patch_h, pw=patch_w)

        # Build normalized 2D pixel query grid in [0, 1).
        # Shape: (patch_h, patch_w, 2)
        qh = torch.arange(patch_h, dtype=x.dtype, device=x.device) / patch_h
        qw = torch.arange(patch_w, dtype=x.dtype, device=x.device) / patch_w
        grid_h, grid_w = torch.meshgrid(qh, qw, indexing="ij")
        query = torch.stack([grid_h, grid_w], dim=-1)  # (patch_h, patch_w, 2)

        # Broadcast query to batch: (B*nh*nw, patch_h, patch_w, 2)
        query = query.unsqueeze(0).expand(tokens.shape[0], -1, -1, -1)

        # Concat: (B*nh*nw, patch_h, patch_w, C+2)
        perceiver_input = torch.cat([tokens, query], dim=-1)

        # --- Run shared perceiver ---
        # The perceiver cross-attends its learned latents to the (patch_h, patch_w)
        # grid of (C+2)-dim tokens, then mean-pools to (out_channels,).
        # (B*nh*nw, patch_h, patch_w, C+2) -> (B*nh*nw, out_channels)
        out = self.perceiver(perceiver_input)

        # --- Reshape back to latent grid ---
        out = rearrange(out, "(b nh nw) c -> b c nh nw", b=B, nh=nh, nw=nw)

        return out
