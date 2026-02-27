# Perceiver-based decoder, complementary to encoder.py

import torch
from aurora.model.fourier import pos_expansion, scale_expansion
from aurora.model.posencoding import pos_scale_enc
from einops import rearrange
from jaxtyping import Float
from torch import nn

from ocean_emulators.constants import Lat, Lon
from ocean_emulators.models.modules.encoder import patch_from


class PerceiverDecoder(nn.Module):
    """A PerceiverIO-based[0] decoder that maps latent patch tokens to full-resolution output.

    Unlike the encoder's regular Perceiver (which compresses spatial input into
    a fixed-size latent), the decoder uses **PerceiverIO**[3] — a variant with
    an explicit query mechanism.  Queries represent output pixel positions, and
    the PerceiverIO cross-attends from those queries to the encoded latent
    representation, producing one prediction per query.

    For each patch position on the latent grid, the decoder:

    1. Adds Aurora-style pos/scale encoding to the latent vector (telling the
       model *where on the globe* this patch is).
    2. Passes the encoded latent as **data** to the PerceiverIO — the single
       token that the model's internal latents cross-attend to during encoding.
    3. Builds normalized 2D pixel-position **queries** for every output pixel
       within the patch, embeds them via a learned linear layer into
       ``queries_dim``, and feeds them to the PerceiverIO's decoder head.
    4. The PerceiverIO's decoder cross-attends from the queries to the encoded
       latents, producing ``(num_pixels, out_channels)`` — one output per pixel.
    5. The per-pixel outputs are reassembled into ``(B, out_channels, H, W)``.

    **Multi-scale windowing**: At higher resolutions, each patch contains more
    pixels and therefore more queries.  To keep per-call cost bounded, when the
    number of pixels exceeds ``window_size`` the decoder splits queries into
    fixed-size windows and calls the PerceiverIO once per window.  Each window
    re-encodes the same latent data (cheap — it's a single token) but only
    decodes its subset of pixel queries.

    Because pixel queries are normalized to ``[0, 1)``, the same PerceiverIO
    generalizes across resolutions: a 1-degree patch queries a coarse grid
    while a 0.25-degree patch queries a finer grid over the same coordinate
    space.

    Args:
        in_channels: Number of input channels from the processor.
        out_channels: Number of output channels per pixel.
        patch_extent: Spatial extent of each patch in degrees (lat, lon).
            Used for computing positional and scale encodings.
        queries_dim: Embedding dimension for pixel-position queries.
        perceiver_io: A PerceiverIO module.  ``dim`` must equal ``in_channels``,
            ``queries_dim`` must match this decoder's ``queries_dim``, and
            ``logits_dim`` must equal ``out_channels``.
        window_size: Maximum number of pixel queries per PerceiverIO call.
            If ``None``, all pixels in a patch are decoded in one call.
            Set this to cap memory/compute at high resolutions.

    References:
        [0]: https://github.com/lucidrains/perceiver-pytorch
        [1]: https://ar5iv.labs.arxiv.org/html/2405.13063#A2.SS4
        [2]: https://ar5iv.labs.arxiv.org/html/2309.16588
        [3]: https://ar5iv.labs.arxiv.org/html/2107.14795
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_extent: tuple[float, float],
        queries_dim: int,
        perceiver_io: nn.Module,
        window_size: int | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_extent = patch_extent
        self.window_size = window_size

        # TODO(#451): The input to these position and scale linear units could be a hparam.
        # Same pos/scale linear layers as the encoder, but applied *before* the
        # perceiver (the encoder applies them after).
        self.pos_embed = nn.Linear(in_channels, in_channels)
        self.scale_embed = nn.Linear(in_channels, in_channels)

        # Embed 2D pixel coordinates into queries_dim for the PerceiverIO decoder head.
        self.query_embed = nn.Linear(2, queries_dim)

        self.perceiver_io = perceiver_io

    def forward(
        self,
        x: Float[torch.Tensor, "batch channels nh nw"],
        resolution: tuple[Lat, Lon],
    ) -> Float[torch.Tensor, "batch {self.out_channels} H W"]:
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

        # --- Add pos/scale encoding to latent tokens (before perceiver, unlike encoder) ---
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

        # --- Data for PerceiverIO: one latent token per patch ---
        # (B, nh*nw, C) -> (B*nh*nw, 1, C)
        data = rearrange(tokens, "b (nh nw) c -> (b nh nw) 1 c", nh=nh, nw=nw)

        # --- Build pixel-position queries ---
        # Normalized 2D coords in [0, 1), embedded into queries_dim.
        qh = torch.arange(patch_h, dtype=x.dtype, device=x.device) / patch_h
        qw = torch.arange(patch_w, dtype=x.dtype, device=x.device) / patch_w
        grid_h, grid_w = torch.meshgrid(qh, qw, indexing="ij")
        coords = torch.stack([grid_h, grid_w], dim=-1)  # (patch_h, patch_w, 2)
        coords = rearrange(coords, "ph pw d -> (ph pw) d")  # (num_pixels, 2)
        queries = self.query_embed(coords)  # (num_pixels, queries_dim)

        # --- Decode via PerceiverIO with optional windowing ---
        out = self._decode(data, queries)  # (B*nh*nw, num_pixels, out_channels)

        # --- Reassemble into full-resolution output ---
        out = rearrange(
            out,
            "(b nh nw) (ph pw) c -> b c (nh ph) (nw pw)",
            b=B,
            nh=nh,
            nw=nw,
            ph=patch_h,
            pw=patch_w,
        )

        return out

    def _decode(
        self,
        data: torch.Tensor,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        """Run PerceiverIO, windowing over queries if they exceed window_size.

        Args:
            data: Encoded latent tokens, shape ``(batch, 1, C)``.
            queries: Embedded pixel queries, shape ``(num_pixels, queries_dim)``.
                Shared across the batch (PerceiverIO broadcasts internally).

        Returns:
            Output predictions, shape ``(batch, num_pixels, out_channels)``.
        """
        num_pixels = queries.shape[0]

        if self.window_size is None or num_pixels <= self.window_size:
            return self.perceiver_io(data, queries=queries)

        # Split queries into fixed-size windows to cap memory at high resolutions.
        # Each window re-encodes the same data (one latent token, cheap) but only
        # decodes its subset of pixel queries.
        chunks = []
        for i in range(0, num_pixels, self.window_size):
            q_window = queries[i : i + self.window_size]
            chunks.append(self.perceiver_io(data, queries=q_window))
        return torch.cat(chunks, dim=1)
