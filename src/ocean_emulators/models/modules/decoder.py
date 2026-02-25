# Mirrors the encoder structure in encoder.py
# Sources:
# - https://ar5iv.labs.arxiv.org/html/2405.13063 (Aurora paper, Appendix B.3: 3D Perceiver Decoder)
# - https://github.com/lucidrains/perceiver-pytorch

import torch
from aurora.model.fourier import pos_expansion, scale_expansion
from aurora.model.posencoding import pos_scale_enc
from einops import rearrange
from jaxtyping import Float
from torch import nn

from ocean_emulators.constants import Lat, Lon
from ocean_emulators.models.modules.encoder import patch_from


class PerceiverDecoder(nn.Module):
    """A perceiver-based decoder that mirrors the PerceiverEncoder.

    While the encoder compresses per-patch physical data into latent tokens via
    cross-attention (many input tokens -> few latent queries), this decoder
    expands the processor's latent grid back to output channels via cross-attention
    (latent context -> output position queries).

    The decoder operates on the **full latent grid** rather than per-patch because:
    - Per-patch decoding with a single context token degenerates cross-attention
      into a linear projection (no attention dynamics).
    - The latent grid is small (~1080 tokens for all resolutions with default
      patch extents), making full-grid attention very efficient.
    - Global attention enables long-range spatial dependencies in the decoder.

    Architecture per forward pass:
        1. Project processor output to decoder latent dimension
        2. Add positional + scale encoding to latent context (mirrors encoder)
        3. Cross-attention: learned output queries attend to latent context
        4. Self-attention: output queries refine via mutual attention
        5. Linear projection to output channels
        6. Reshape to spatial grid (B, out_channels, h, w)

    The output is then passed through the existing ``unpatch`` system in FOMO
    for pixel-level expansion back to the original spatial resolution.

    Args:
        in_channels: Number of input channels from the processor.
        out_channels: Number of output channels per spatial position.
        patch_extent: Spatial extent of each patch in degrees (lat, lon).
            Used for computing positional and scale encodings.
        perceiver: The underlying perceiver decoder.
        latent_dim: Dimension of the decoder's internal latent space.

    References:
        [1]: https://ar5iv.labs.arxiv.org/html/2405.13063#A2.SS4
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_extent: tuple[float, float],
        perceiver: nn.Module,
        latent_dim: int = 128,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.patch_extent = patch_extent

        # TODO(claude) I don't think we need to project to a latent space; the output
        #  of the processor will be a latent space.
        # Project processor output channels to decoder latent dimension
        self.input_proj = nn.Linear(in_channels, latent_dim)

        # Positional and scale encoding (mirrors encoder's post-perceiver encoding)
        self.pos_embed = nn.Linear(latent_dim, latent_dim)
        self.scale_embed = nn.Linear(latent_dim, latent_dim)

        self.perceiver = perceiver

        # Final projection from latent_dim to out_channels
        self.to_out = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, out_channels),
        )

    def forward(
        self, x: Float[torch.Tensor, "batch channels h w"], resolution: tuple[Lat, Lon]
    ) -> Float[torch.Tensor, "batch {self.out_channels} h w"]:
        B, C, h, w = x.shape
        lat, lon = resolution

        # Compute patch sizes from the original resolution for pos/scale encoding.
        H, W = len(lat), len(lon)
        patch_h, patch_w = patch_from(self.patch_extent, H, W)

        # Reshape processor output to token sequence: (B, h*w, C)
        context = rearrange(x, "b c h w -> b (h w) c")

        # Project to decoder latent dimension
        context = self.input_proj(context)  # (B, h*w, latent_dim)

        # Add positional + scale encoding (mirrors encoder's post-perceiver step)
        pos_encode, scale_encode = pos_scale_enc(
            context.shape[-1],  # latent_dim
            lat,
            lon,
            (patch_h, patch_w),
            pos_expansion=pos_expansion,
            scale_expansion=scale_expansion,
        )
        pos_encoding = self.pos_embed(
            pos_encode.to(dtype=context.dtype, device=context.device)
        ).unsqueeze(0)
        scale_encoding = self.scale_embed(
            scale_encode.to(dtype=context.dtype, device=context.device)
        ).unsqueeze(0)
        context = context + pos_encoding + scale_encoding

        # # Initialize output queries: (B, num_queries, latent_dim)
        # queries = self.queries.unsqueeze(0).expand(B, -1, -1)
        #
        # # Cross-attention + self-attention layers
        # for cross_attn, cross_ff, self_attns in self.layers:  # type: ignore
        #     queries = cross_attn(queries, context=context) + queries  # type: ignore
        #     queries = cross_ff(queries) + queries  # type: ignore
        #
        #     for self_attn, self_ff in self_attns:  # type: ignore
        #         queries = self_attn(queries) + queries
        #         queries = self_ff(queries) + queries
        #

        queries = self.perceiver(context)

        # Project to output channels: (B, h*w, out_channels)
        out = self.to_out(queries)

        # Reshape back to spatial grid
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w)

        return out
