# Perceiver-based decoder, complementary to encoder.py

import torch
from aurora.model.fourier import pos_expansion, scale_expansion
from aurora.model.posencoding import pos_scale_enc
from einops import rearrange
from jaxtyping import Float
from torch import nn

from ocean_emulators.constants import Lat, Lon


class PerceiverDecoder(nn.Module):
    """A PerceiverIO-based decoder that maps a latent patch grid to full-resolution output.

    All ``nh * nw`` pos/scale-encoded latent tokens are passed as **data** to
    the PerceiverIO[3], and every output pixel position is a **query**.  Each
    query cross-attends to the full latent representation, giving it global
    spatial context — pixels near patch boundaries can attend to neighboring
    patches, and the model can learn smooth inter-patch transitions.

    Concretely:

    1. Add Aurora-style pos/scale encoding to the ``nh * nw`` latent tokens
       (telling the model *where on the globe* each patch is).
    2. Pass all encoded latents as **data** to the PerceiverIO:
       ``(B, nh * nw, C)``.
    3. Build normalized 2D **queries** for every output pixel ``(i/H, j/W)``
       in ``[0, 1)``, embed them via a learned linear layer, and feed them
       to the PerceiverIO decoder head.
    4. The PerceiverIO's decoder cross-attends from queries to the internal
       latents (which themselves cross-attended to all patch tokens),
       producing ``(B, H * W, out_channels)``.
    5. Reshape to ``(B, out_channels, H, W)``.

    **Multi-scale windowing**: At higher resolutions ``H * W`` grows, making
    the full query set expensive.  When ``window_size`` is set, the decoder
    splits pixel queries into fixed-size chunks and calls the PerceiverIO once
    per chunk.  Each call re-encodes the same latent data (the ``nh * nw``
    tokens — cheap for small latent grids) and only decodes its subset of
    pixel queries, keeping per-call cost bounded.

    Because pixel queries are normalized to ``[0, 1)``, the same PerceiverIO
    generalizes across resolutions.

    Args:
        in_channels: Number of input channels from the processor.
        out_channels: Number of output channels per pixel.
        patch_extent: Spatial extent of each patch in degrees (lat, lon).
            Used for computing positional and scale encodings on latent tokens.
        queries_dim: Embedding dimension for pixel-position queries.
        perceiver_io: A PerceiverIO module.  ``dim`` must equal ``in_channels``,
            ``queries_dim`` must match this decoder's ``queries_dim``, and
            ``logits_dim`` must equal ``out_channels``.
        window_size: Maximum number of pixel queries per PerceiverIO call.
            If ``None``, all ``H * W`` pixels are decoded in one call.
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
        B, C, nh, nw = x.shape
        lat, lon = resolution

        H, W = len(lat), len(lon)

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
        # tokens: (B, nh*nw, C) — full latent grid as data for PerceiverIO.

        # --- Build global pixel-position queries ---
        # Normalized 2D coords in [0, 1) for every output pixel.
        qh = torch.arange(H, dtype=x.dtype, device=x.device) / H
        qw = torch.arange(W, dtype=x.dtype, device=x.device) / W
        grid_h, grid_w = torch.meshgrid(qh, qw, indexing="ij")
        coords = torch.stack([grid_h, grid_w], dim=-1)  # (H, W, 2)
        coords = rearrange(coords, "h w d -> (h w) d")  # (H*W, 2)
        queries = self.query_embed(coords)  # (H*W, queries_dim)

        # --- Decode via PerceiverIO with optional windowing ---
        out = self._decode(tokens, queries)  # (B, H*W, out_channels)

        # --- Reshape to spatial ---
        out = rearrange(out, "b (h w) c -> b c h w", h=H, w=W)

        return out

    def _decode(
        self,
        data: torch.Tensor,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        """Run PerceiverIO, windowing over queries if they exceed window_size.

        Args:
            data: Pos/scale-encoded latent tokens, shape ``(B, nh*nw, C)``.
            queries: Embedded pixel queries, shape ``(H*W, queries_dim)``.
                Shared across the batch (PerceiverIO broadcasts internally).

        Returns:
            Output predictions, shape ``(B, H*W, out_channels)``.
        """
        num_pixels = queries.shape[0]

        if self.window_size is None or num_pixels <= self.window_size:
            return self.perceiver_io(data, queries=queries)

        # Split queries into fixed-size windows to cap memory at high resolutions.
        # Each window re-encodes the same latent data and only decodes its
        # subset of pixel queries.
        chunks = []
        for i in range(0, num_pixels, self.window_size):
            q_window = queries[i : i + self.window_size]
            chunks.append(self.perceiver_io(data, queries=q_window))
        return torch.cat(chunks, dim=1)
