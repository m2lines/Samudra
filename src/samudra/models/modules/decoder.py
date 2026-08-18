# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

# Perceiver-based decoder, complementary to encoder.py

import torch
import torch.nn.functional as F
from aurora.model.fourier import pos_expansion, scale_expansion
from aurora.model.posencoding import pos_scale_enc
from einops import rearrange
from jaxtyping import Float
from perceiver_pytorch.perceiver_io import Attention, FeedForward, PreNorm
from torch import nn

from samudra.constants import Lat, Lon
from samudra.models.modules.augment_input import make_3d_coordinate_grid
from samudra.models.modules.encoder import patch_from


class DirectCrossAttentionIO(nn.Module):
    """Apply the Perceiver IO decode stage directly to processor tokens.

    ``SamudraMulti`` already has a Perceiver encoder followed by a spatial latent
    processor. Running a complete ``PerceiverIO`` inside the decoder constructs a
    second learned latent array and first compresses the processor tokens into it.
    This module keeps the Perceiver IO output-query interface but removes that
    redundant encode/process stage: output queries cross-attend directly to the
    processor tokens.

    The query residual is intentional. Without it, attention over a single context
    token is independent of the query because its softmax has length one. Keeping
    the residual matches the general Perceiver cross-attention block and guarantees
    that spatial output queries retain a path to the decoded representation.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        queries_dim: int,
        output_dim: int,
        heads: int,
        dim_head: int,
    ) -> None:
        super().__init__()
        if heads < 1:
            raise ValueError("heads must be positive.")
        if dim_head < 1:
            raise ValueError("dim_head must be positive.")
        self.cross_attn = PreNorm(
            queries_dim,
            Attention(
                queries_dim,
                input_dim,
                heads=heads,
                dim_head=dim_head,
            ),
            context_dim=input_dim,
        )
        self.feed_forward = PreNorm(queries_dim, FeedForward(queries_dim))
        self.to_logits = nn.Linear(queries_dim, output_dim)

    def forward(
        self,
        data: torch.Tensor,
        *,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        if queries.ndim == 2:
            queries = queries.unsqueeze(0).expand(data.shape[0], -1, -1)
        if queries.ndim != 3:
            raise ValueError(
                "queries must have shape [outputs, channels] or "
                f"[batch, outputs, channels], got {tuple(queries.shape)}."
            )
        if queries.shape[0] != data.shape[0]:
            raise ValueError(
                "Batched queries and data must have the same batch size, got "
                f"{queries.shape[0]} and {data.shape[0]}."
            )

        decoded = queries + self.cross_attn(queries, context=data)
        decoded = decoded + self.feed_forward(decoded)
        return self.to_logits(decoded)


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
    3. Build 3D unit-sphere **queries** ``(x, y, z)`` for every output pixel
       from its lat/lon, embed them via a learned linear layer, and feed
       them to the PerceiverIO decoder head.
    4. Inside the PerceiverIO:
       a. Internal latents cross-attend to the ``nh * nw`` data tokens.
       b. The latents refine through several rounds of self-attention.
       c. A final cross-attention maps from queries to the refined latents,
          producing ``(B, H * W, out_channels)``.
    5. Reshape to ``(B, out_channels, H, W)``.

    **Spatial windowing**: When ``window_patches`` is set, the latent grid
    must be evenly divisible by ``window_patches``.  The grid is padded —
    circular along longitude (so windows near lon=0 see context from
    lon≈360) and constant-zero along latitude (poles are true boundaries)
    — then ``Tensor.unfold`` extracts fixed-size overlapping windows.
    Each block's PerceiverIO call receives the local data context plus
    the corresponding pixel queries.  Setting ``context_patches=None``
    gives each window full access to all latent tokens (windowed queries,
    global data).

    Because pixel queries are unit-sphere coordinates — continuous values
    determined by lat/lon, not grid indices — the same PerceiverIO
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
        [0]: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
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
        window_patches: int | None,
        context_patches: int | None,
        output_overlap_patches: int = 0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_extent = patch_extent
        if window_patches is None and context_patches is not None:
            raise ValueError(
                "window_patches must be set in order for context_patches to be set."
            )
        self.window_patches = window_patches
        self.context_patches = context_patches
        if output_overlap_patches < 0:
            raise ValueError("output_overlap_patches must be non-negative.")
        if window_patches is None and output_overlap_patches:
            raise ValueError(
                "window_patches must be set when output_overlap_patches is nonzero."
            )
        if window_patches is not None and output_overlap_patches > window_patches // 2:
            raise ValueError(
                "output_overlap_patches must not exceed half of window_patches."
            )
        self.output_overlap_patches = output_overlap_patches

        # TODO(#451): The input to these position and scale linear units could be a hparam.
        # Same pos/scale linear layers as the encoder, but applied *before* the
        # perceiver (the encoder applies them after).
        self.pos_embed = nn.Linear(in_channels, in_channels)
        self.scale_embed = nn.Linear(in_channels, in_channels)

        # Embed 3D unit-sphere coordinates into queries_dim for the PerceiverIO decoder head.
        self.query_embed = nn.Linear(3, queries_dim)

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

        pos_patch_h, pos_patch_w = patch_from(self.patch_extent, H, W)

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
        # 3D unit-sphere coordinates for every output pixel.
        coords = make_3d_coordinate_grid(lat, lon)  # (3, H, W)
        coords = rearrange(coords, "d h w -> h w d").to(
            dtype=x.dtype, device=x.device
        )  # (H, W, 3)
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
        """Decode a latent patch grid into full-resolution pixel output.

        Without windowing, every pixel query attends to every latent token
        (global attention).  With windowing, the grid is split into spatial
        blocks so each PerceiverIO call only covers a local neighborhood,
        keeping cost bounded for large latent grids.
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

        assert nh % wp == 0 and nw % wp == 0, (
            f"Latent grid ({nh}, {nw}) must be divisible by window_patches={wp}"
        )

        if self.output_overlap_patches:
            return self._decode_overlapping(data_grid, queries_grid, patch_h, patch_w)

        n_blocks_h = nh // wp
        n_blocks_w = nw // wp
        block_ph = wp * patch_h  # pixel height per query block
        block_pw = wp * patch_w  # pixel width per query block

        # --- Prepare data windows ---
        if cp is None:
            # Full context: every window sees all latent tokens.
            full_data = rearrange(data_grid, "b nh nw c -> b (nh nw) c")
        elif cp == 0:
            # No context padding — unfold with exact window size.
            data = rearrange(data_grid, "b nh nw c -> b c nh nw")
            data_windows = data.unfold(2, wp, wp).unfold(3, wp, wp)
        else:
            # Pad: circular along longitude (last dim), zero along latitude.
            data = rearrange(data_grid, "b nh nw c -> b c nh nw")
            data = F.pad(data, (cp, cp, 0, 0), mode="circular")
            data = F.pad(data, (0, 0, cp, cp), mode="constant", value=0)
            win_size_h = wp + 2 * cp
            win_size_w = wp + 2 * cp
            data_windows = data.unfold(2, win_size_h, wp).unfold(3, win_size_w, wp)
        # data_windows shape (when cp is not None):
        #   (B, C, n_blocks_h, n_blocks_w, win_h, win_w)

        # --- Decode each spatial block ---
        out = data_grid.new_zeros(B, H, W, self.out_channels)

        for bi in range(n_blocks_h):
            for bj in range(n_blocks_w):
                if cp is None:
                    local_data = full_data
                else:
                    local_data = rearrange(
                        data_windows[:, :, bi, bj], "b c h w -> b (h w) c"
                    )

                qi_start = bi * block_ph
                qj_start = bj * block_pw
                local_queries = queries_grid[
                    qi_start : qi_start + block_ph,
                    qj_start : qj_start + block_pw,
                ]
                local_queries = rearrange(local_queries, "h w d -> (h w) d")

                local_out = self.perceiver_io(local_data, queries=local_queries)
                local_out = rearrange(
                    local_out, "b (h w) c -> b h w c", h=block_ph, w=block_pw
                )
                out[
                    :,
                    qi_start : qi_start + block_ph,
                    qj_start : qj_start + block_pw,
                    :,
                ] = local_out

        return rearrange(out, "b h w c -> b c h w")

    def _decode_overlapping(
        self,
        data_grid: Float[torch.Tensor, "batch nh nw channels"],
        queries_grid: Float[torch.Tensor, "H W queries_dim"],
        patch_h: int,
        patch_w: int,
    ) -> Float[torch.Tensor, "batch {self.out_channels} H W"]:
        """Decode query halos and combine repeated predictions smoothly.

        Input context remains centered on each ``window_patches`` core. The
        overlap changes only output support: pixels around every core are
        predicted by neighboring windows and cosine blended. Longitude wraps
        periodically; latitude halos stop at the physical domain boundaries.
        """
        B, nh, nw, _ = data_grid.shape
        H, W, _ = queries_grid.shape
        wp = self.window_patches
        cp = self.context_patches
        op = self.output_overlap_patches
        assert wp is not None and op > 0
        assert wp + 2 * op <= nw, (
            "An overlapping longitude window must not wrap over itself."
        )

        weighted = data_grid.new_zeros(B, H, W, self.out_channels)
        weight_sum = data_grid.new_zeros(H, W, 1)
        n_blocks_h = nh // wp
        n_blocks_w = nw // wp
        halo_h = op * patch_h
        halo_w = op * patch_w

        if cp is None:
            full_data = rearrange(data_grid, "b nh nw c -> b (nh nw) c")
        elif cp == 0:
            data = rearrange(data_grid, "b nh nw c -> b c nh nw")
            data_windows = data.unfold(2, wp, wp).unfold(3, wp, wp)
        else:
            data = rearrange(data_grid, "b nh nw c -> b c nh nw")
            data = F.pad(data, (cp, cp, 0, 0), mode="circular")
            data = F.pad(data, (0, 0, cp, cp), mode="constant", value=0)
            window_size = wp + 2 * cp
            data_windows = data.unfold(2, window_size, wp).unfold(3, window_size, wp)

        for bi in range(n_blocks_h):
            core_patch_i0 = bi * wp
            core_patch_i1 = (bi + 1) * wp
            query_i0 = max(0, (core_patch_i0 - op) * patch_h)
            query_i1 = min(H, (core_patch_i1 + op) * patch_h)
            lat_indices = torch.arange(query_i0, query_i1, device=data_grid.device)
            lat_weights = self._overlap_weights(
                len(lat_indices),
                halo_h,
                fade_start=bi > 0,
                fade_end=bi + 1 < n_blocks_h,
                device=data_grid.device,
                dtype=data_grid.dtype,
            )

            for bj in range(n_blocks_w):
                core_patch_j0 = bj * wp
                core_patch_j1 = (bj + 1) * wp
                query_j0 = (core_patch_j0 - op) * patch_w
                query_j1 = (core_patch_j1 + op) * patch_w
                lon_indices = torch.arange(
                    query_j0, query_j1, device=data_grid.device
                ).remainder(W)

                if cp is None:
                    local_data = full_data
                else:
                    local_data = rearrange(
                        data_windows[:, :, bi, bj], "b c h w -> b (h w) c"
                    )

                local_queries = queries_grid.index_select(0, lat_indices)
                local_queries = local_queries.index_select(1, lon_indices)
                local_queries = rearrange(local_queries, "h w d -> (h w) d")
                local_out = self.perceiver_io(local_data, queries=local_queries)
                local_out = rearrange(
                    local_out,
                    "b (h w) c -> b h w c",
                    h=len(lat_indices),
                    w=len(lon_indices),
                )

                lon_weights = self._overlap_weights(
                    len(lon_indices),
                    halo_w,
                    fade_start=True,
                    fade_end=True,
                    device=data_grid.device,
                    dtype=data_grid.dtype,
                )
                weights = lat_weights[:, None] * lon_weights[None, :]
                weighted[:, lat_indices[:, None], lon_indices[None, :], :] += (
                    local_out * weights[None, :, :, None]
                )
                weight_sum[lat_indices[:, None], lon_indices[None, :], :] += weights[
                    :, :, None
                ]

        if not torch.all(weight_sum > 0):
            raise RuntimeError("Overlapping decoder left output pixels uncovered.")
        return rearrange(weighted / weight_sum, "b h w c -> b c h w")

    @staticmethod
    def _overlap_weights(
        length: int,
        halo: int,
        *,
        fade_start: bool,
        fade_end: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        weights = torch.ones(length, device=device, dtype=dtype)
        ramp_length = 2 * halo
        if ramp_length == 0:
            return weights
        if ramp_length > length:
            raise ValueError("Overlap ramp is longer than the decoded window.")
        phase = (torch.arange(ramp_length, device=device, dtype=dtype) + 0.5) / (
            ramp_length
        )
        ramp = 0.5 - 0.5 * torch.cos(torch.pi * phase)
        if fade_start:
            weights[:ramp_length] = ramp
        if fade_end:
            weights[-ramp_length:] = ramp.flip(0)
        return weights
