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

    After decoding, the output is then passed through the existing ``unpatch``
    system in FOMO for pixel-level expansion back to the original spatial
    resolution.

    Args:
        in_channels: Number of input channels from the processor.
        out_channels: Number of output channels per spatial position.
        patch_extent: Spatial extent of each patch in degrees (lat, lon).
            Used for computing positional and scale encodings.
        perceiver: The underlying perceiver decoder. Must project to
            ``out_channels`` internally (e.g. via ``num_classes=out_channels``
            in the NaivePerceiver).

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
        self.out_channels = out_channels
        self.patch_extent = patch_extent

        # Positional and scale encoding (mirrors encoder's post-perceiver encoding)
        self.pos_embed = nn.Linear(in_channels, in_channels)
        self.scale_embed = nn.Linear(in_channels, in_channels)

        self.perceiver = perceiver

    def forward(
        self, x: Float[torch.Tensor, "batch channels h w"], resolution: tuple[Lat, Lon]
    ) -> Float[torch.Tensor, "batch {self.out_channels} h w"]:
        B, C, h, w = x.shape  # h = input H // patch_h; w = input W // patch_w
        lat, lon = resolution

        # Derive the effective patch size from the original resolution and the
        # decoder's spatial grid.  In the full FOMO pipeline the processor output
        # (h, w) equals (H // patch_h, W // patch_w), but we compute the patch
        # size from the tensor itself so the pos/scale encodings always produce
        # exactly h*w tokens — even in unit tests where h, w may differ from the
        # patch-reduced grid.
        H, W = len(lat), len(lon)  # Original (physical) resolution.
        patch_h, patch_w = H // h, W // w

        # Reshape processor output to token sequence: (B, h*w, C)
        context = rearrange(x, "b c h w -> b (h w) c")

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

        context = rearrange(context, "b (h w) c -> (b h w) c", h=h, w=w)
        # We add these middle singleton dimensions to make this perceiver consistent with the encoder's perceiver.
        # It assumes that `input_axis=2`.
        context = context.unsqueeze(1).unsqueeze(1)

        # (B*h*w, 1, 1, channels) -> (B*h*w, out_channels)
        out = self.perceiver(context)

        # Reshape back to spatial grid
        out = rearrange(out, "(b h w) c -> b c h w", h=h, w=w)

        return out
