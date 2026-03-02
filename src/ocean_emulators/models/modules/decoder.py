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
    the PerceiverIO[2], and every output pixel position is a **query**.  Each
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

    **Spatial windowing**: When ``window_patches`` is set, the decoder tiles
    the output grid into spatial blocks, each covering ``window_patches``
    patches along each axis.  For each block, only the overlapping latent
    tokens — plus ``context_patches`` extra rings of neighbors — are passed
    as data.  This bounds both query count and data count per PerceiverIO
    call, keeping cost manageable even when the latent grid is large (i.e.
    fine ``patch_extent``).  Setting ``context_patches=None`` gives each
    window full access to all latent tokens (windowed queries, global data).

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
        window_patches: Side length (in patches) of each spatial decode window.
            If ``None``, all patches are used globally (no windowing).
            E.g. ``window_patches=8`` means each PerceiverIO call covers an
            8x8 block of patches.
        context_patches: Number of extra patch rings around each window to
            include as data context.  Only used when ``window_patches`` is set.
            Default 1 gives each window one ring of neighboring patches beyond
            its own block.  ``None`` means full context — every window sees all
            latent tokens (windowed queries but global data attention).

    References:
        [0]: https://github.com/lucidrains/perceiver-pytorch
        [1]: https://ar5iv.labs.arxiv.org/html/2405.13063#A2.SS4
        [2]: https://ar5iv.labs.arxiv.org/html/2107.14795
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_extent: tuple[float, float],
        queries_dim: int,
        perceiver_io: nn.Module,
        window_patches: int | None = None,
        context_patches: int | None = 1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_extent = patch_extent
        self.window_patches = window_patches
        self.context_patches = context_patches

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

        # --- Build global pixel-position queries ---
        # Normalized 2D coords in [0, 1) for every output pixel.
        qh = torch.arange(H, dtype=x.dtype, device=x.device) / H
        qw = torch.arange(W, dtype=x.dtype, device=x.device) / W
        grid_h, grid_w = torch.meshgrid(qh, qw, indexing="ij")
        coords = torch.stack([grid_h, grid_w], dim=-1)  # (H, W, 2)
        queries = self.query_embed(
            rearrange(coords, "h w d -> (h w) d")
        )  # (H*W, queries_dim)
        queries = rearrange(
            queries, "(h w) d -> h w d", h=H, w=W
        )  # (H, W, queries_dim)

        # --- Decode via PerceiverIO with optional spatial windowing ---
        data_grid = rearrange(tokens, "b (nh nw) c -> b nh nw c", nh=nh, nw=nw)
        out = self._decode(data_grid, queries, pos_patch_h, pos_patch_w)

        return out

    def _decode(
        self,
        data_grid: Float[torch.Tensor, "batch nh nw channels"],
        queries_grid: Float[torch.Tensor, "H W queries_dim"],
        patch_h: int,
        patch_w: int,
    ) -> Float[torch.Tensor, "batch {self.out_channels} H W"]:
        """Run PerceiverIO, optionally tiling into spatial windows.

        When ``window_patches`` is None, all data and queries are passed in one
        call (global attention).  Otherwise, the output grid is tiled into
        spatial blocks and each block attends only to nearby latent tokens.

        Note: the windowed loop below is structurally equivalent to
        ``nn.Unfold`` / ``im2col`` — strided extraction of overlapping 2D
        patches.  We use explicit indexing instead because (1) ``unfold``
        requires pre-padding and produces fixed-size windows, adding edge-
        handling complexity for grids not evenly divisible by
        ``window_patches``, and (2) the bottleneck is the PerceiverIO forward
        pass per window, not the Python index arithmetic, so there is no
        performance benefit to fusing the extraction.
        """
        B, nh, nw, C = data_grid.shape
        H, W, _ = queries_grid.shape

        if self.window_patches is None:
            data = rearrange(data_grid, "b nh nw c -> b (nh nw) c")
            queries = rearrange(queries_grid, "h w d -> (h w) d")
            out = self.perceiver_io(data, queries=queries)  # (B, H*W, out_channels)
            return rearrange(out, "b (h w) c -> b c h w", h=H, w=W)

        wp = self.window_patches
        cp = self.context_patches

        out = data_grid.new_zeros(B, H, W, self.out_channels)

        # Flatten full data once when context_patches is None (full context).
        full_data = (
            rearrange(data_grid, "b nh nw c -> b (nh nw) c") if cp is None else None
        )

        for pi in range(0, nh, wp):
            for pj in range(0, nw, wp):
                pi_end = min(pi + wp, nh)
                pj_end = min(pj + wp, nw)

                if cp is None:
                    # Full context: every window sees all latent tokens.
                    local_data = full_data
                else:
                    # Expand data region by context_patches, clamped to grid bounds.
                    di_start = max(pi - cp, 0)
                    di_end = min(pi_end + cp, nh)
                    dj_start = max(pj - cp, 0)
                    dj_end = min(pj_end + cp, nw)

                    local_data = data_grid[:, di_start:di_end, dj_start:dj_end, :]
                    local_data = rearrange(local_data, "b h w c -> b (h w) c")

                # Pixel region covered by this patch block.
                qi_start = pi * patch_h
                qi_end = pi_end * patch_h
                qj_start = pj * patch_w
                qj_end = pj_end * patch_w

                local_queries = queries_grid[qi_start:qi_end, qj_start:qj_end, :]
                local_queries = rearrange(local_queries, "h w d -> (h w) d")

                local_out = self.perceiver_io(local_data, queries=local_queries)
                qh_size = qi_end - qi_start
                qw_size = qj_end - qj_start
                local_out = rearrange(
                    local_out, "b (h w) c -> b h w c", h=qh_size, w=qw_size
                )

                out[:, qi_start:qi_end, qj_start:qj_end, :] = local_out

        return rearrange(out, "b h w c -> b c h w")
