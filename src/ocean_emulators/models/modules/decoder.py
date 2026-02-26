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

    The encoder compresses each patch of physical data into a single latent
    token via perceiver cross-attention. This decoder inverts that: for each
    latent-grid position it runs a perceiver that maps one token back to
    ``out_channels``, then FOMO's ``unpatch`` layer expands each position to
    full patch resolution.

    Positional and scale encodings (Aurora-style [1]) are added to the latent
    tokens before decoding, mirroring the encoder's post-perceiver step.

    Args:
        in_channels: Number of input channels from the processor.
        out_channels: Number of output channels per spatial position.
        patch_extent: Spatial extent of each patch in degrees (lat, lon).
            Used for computing positional and scale encodings.
        perceiver: The underlying perceiver module. Must project to
            ``out_channels`` internally (e.g. ``num_classes=out_channels``).

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
        self,
        x: Float[torch.Tensor, "batch channels nh nw"],
        resolution: tuple[Lat, Lon],
    ) -> Float[torch.Tensor, "batch {self.out_channels} nh nw"]:
        # nh, nw: number of patches along height and width (the latent grid dims).
        # Not to be confused with patch_h, patch_w which are each patch's pixel size.
        B, C, nh, nw = x.shape
        lat, lon = resolution

        # Derive the effective patch size from the original resolution and the
        # decoder's spatial grid.  In the full FOMO pipeline the processor output
        # (nh, nw) equals (H // patch_h, W // patch_w), but we compute the patch
        # size from the tensor itself so the pos/scale encodings always produce
        # exactly nh*nw tokens — even in unit tests where nh, nw may differ from
        # the patch-reduced grid.
        H, W = len(lat), len(lon)  # Original (physical) resolution.
        patch_h, patch_w = H // nh, W // nw

        # Reshape processor output to token sequence: (B, nh*nw, C)
        context = rearrange(x, "b c nh nw -> b (nh nw) c")

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

        context = rearrange(context, "b (nh nw) c -> (b nh nw) c", nh=nh, nw=nw)
        # We add these middle singleton dimensions to make this perceiver consistent with the encoder's perceiver.
        # It assumes that `input_axis=2`.
        context = context.unsqueeze(1).unsqueeze(1)

        # (B*nh*nw, 1, 1, channels) -> (B*nh*nw, out_channels)
        out = self.perceiver(context)

        # Reshape back to spatial grid
        out = rearrange(out, "(b nh nw) c -> b c nh nw", nh=nh, nw=nw)

        return out
