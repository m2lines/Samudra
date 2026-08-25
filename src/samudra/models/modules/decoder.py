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
from torch import nn

from samudra.constants import Lat, Lon
from samudra.models.modules.augment_input import make_3d_coordinate_grid
from samudra.models.modules.blocks import (
    PointwiseLinear,
    ZonallyPeriodicBilinearUpsample,
)
from samudra.models.modules.encoder import patch_from
from samudra.models.modules.perceiver import (
    Attention,
    AttentionBackend,
    FeedForward,
    PreNorm,
)


class DirectCrossAttentionIO(nn.Module):
    """Apply the Perceiver IO decode head directly to processor tokens.

    This is a transformer cross-attention block, not a complete Perceiver by
    itself. It deliberately retains the output-query portion of Perceiver IO
    while omitting a second latent bank and its self-attention stack: the encoder
    already performed Perceiver compression and the spatial processor already
    mixed the resulting tokens.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        queries_dim: int,
        output_dim: int,
        heads: int,
        dim_head: int,
        attention_backend: AttentionBackend = "auto",
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
                backend=attention_backend,
            ),
            context_dim=input_dim,
        )
        self.feed_forward = PreNorm(queries_dim, FeedForward(queries_dim))
        self.to_logits = nn.Linear(queries_dim, output_dim)

    def decode_features(
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
        return decoded + self.feed_forward(decoded)

    def project_features(self, decoded: torch.Tensor) -> torch.Tensor:
        return self.to_logits(decoded)

    def forward(
        self,
        data: torch.Tensor,
        *,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        return self.project_features(self.decode_features(data, queries=queries))


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

    # A window-batched decoder item contains one model sample and one spatial
    # window. Bounding this dimension avoids oversized fused-attention and
    # feed-forward workspaces while retaining far fewer launches than the
    # original one-window-at-a-time implementation.
    _MAX_WINDOW_BATCH_SIZE = 128

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
        processor_conditioning: bool = False,
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
        if processor_conditioning and not isinstance(
            perceiver_io, DirectCrossAttentionIO
        ):
            raise ValueError(
                "processor conditioning requires the direct_cross_attention "
                "decoder architecture."
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
        self.processor_conditioner = (
            PointwiseLinear(in_channels, queries_dim)
            if processor_conditioning
            else None
        )
        self.conditioning_strength = (
            nn.Parameter(torch.zeros(())) if processor_conditioning else None
        )

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

        if self.processor_conditioner is not None:
            if H % nh or W % nw:
                raise ValueError(
                    "Processor conditioning requires integer output scale factors; "
                    f"got processor={(nh, nw)} and output={(H, W)}."
                )
            processor = ZonallyPeriodicBilinearUpsample((H // nh, W // nw))(x)
            conditioning = self.processor_conditioner(processor)
            assert self.conditioning_strength is not None
            out = out + self.conditioning_strength * conditioning
            direct_decoder = self.perceiver_io
            assert isinstance(direct_decoder, DirectCrossAttentionIO)
            out = rearrange(out, "b d h w -> b h w d")
            out = direct_decoder.project_features(out)
            out = rearrange(out, "b h w c -> b c h w")

        return out

    @property
    def decoded_channels(self) -> int:
        return (
            self.query_embed.out_features
            if self.processor_conditioner is not None
            else self.out_channels
        )

    def _decode_queries(
        self,
        data: torch.Tensor,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        if self.processor_conditioner is None:
            return self.perceiver_io(data, queries=queries)
        direct_decoder = self.perceiver_io
        assert isinstance(direct_decoder, DirectCrossAttentionIO)
        return direct_decoder.decode_features(data, queries=queries)

    def _decode_window_batch(
        self,
        data: torch.Tensor,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        """Decode flattened batch-window items in bounded native-kernel batches."""
        if data.shape[0] != queries.shape[0]:
            raise ValueError("Window data and queries must have the same batch size.")
        outputs = [
            self._decode_queries(data_chunk, queries=queries_chunk)
            for data_chunk, queries_chunk in zip(
                data.split(self._MAX_WINDOW_BATCH_SIZE),
                queries.split(self._MAX_WINDOW_BATCH_SIZE),
                strict=True,
            )
        ]
        return torch.cat(outputs)

    @staticmethod
    def _local_data_windows(
        data_grid: torch.Tensor,
        *,
        window_patches: int,
        context_patches: int,
    ) -> torch.Tensor:
        """Return local latent windows as ``[batch, window, token, channel]``."""
        data = rearrange(data_grid, "b nh nw c -> b c nh nw")
        if context_patches:
            data = F.pad(
                data,
                (context_patches, context_patches, 0, 0),
                mode="circular",
            )
            data = F.pad(
                data,
                (0, 0, context_patches, context_patches),
                mode="constant",
                value=0,
            )
        window_size = window_patches + 2 * context_patches
        windows = data.unfold(2, window_size, window_patches).unfold(
            3, window_size, window_patches
        )
        return rearrange(
            windows,
            "b c bh bw wh ww -> b (bh bw) (wh ww) c",
        )

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
            out = self._decode_queries(data, queries)
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

        # Full context cannot be folded into the window batch without
        # materializing the complete latent grid once per window. Retain the
        # memory-bounded path for that uncommon mode; local windows are decoded
        # together below.
        if cp is None:
            full_data = rearrange(data_grid, "b nh nw c -> b (nh nw) c")
            rows = []
            for bi in range(n_blocks_h):
                columns = []
                for bj in range(n_blocks_w):
                    local_queries = queries_grid[
                        bi * block_ph : (bi + 1) * block_ph,
                        bj * block_pw : (bj + 1) * block_pw,
                    ]
                    local_out = self._decode_queries(
                        full_data,
                        rearrange(local_queries, "h w d -> (h w) d"),
                    )
                    columns.append(
                        rearrange(
                            local_out,
                            "b (h w) c -> b h w c",
                            h=block_ph,
                            w=block_pw,
                        )
                    )
                rows.append(torch.cat(columns, dim=2))
            return rearrange(torch.cat(rows, dim=1), "b h w c -> b c h w")

        data_windows = self._local_data_windows(
            data_grid,
            window_patches=wp,
            context_patches=cp,
        )
        query_windows = rearrange(
            queries_grid,
            "(bh ph) (bw pw) d -> (bh bw) (ph pw) d",
            bh=n_blocks_h,
            bw=n_blocks_w,
            ph=block_ph,
            pw=block_pw,
        )
        window_count = n_blocks_h * n_blocks_w
        batched_data = rearrange(data_windows, "b n t c -> (b n) t c")
        batched_queries = (
            query_windows.unsqueeze(0)
            .expand(B, window_count, -1, -1)
            .reshape(B * window_count, block_ph * block_pw, -1)
        )
        decoded = self._decode_window_batch(batched_data, batched_queries)
        return rearrange(
            decoded,
            "(b bh bw) (ph pw) c -> b c (bh ph) (bw pw)",
            b=B,
            bh=n_blocks_h,
            bw=n_blocks_w,
            ph=block_ph,
            pw=block_pw,
        )

    def _decode_overlapping(
        self,
        data_grid: Float[torch.Tensor, "batch nh nw channels"],
        queries_grid: Float[torch.Tensor, "H W queries_dim"],
        patch_h: int,
        patch_w: int,
    ) -> Float[torch.Tensor, "batch channels H W"]:
        """Decode query halos and combine repeated predictions smoothly."""
        batch, nh, nw, _ = data_grid.shape
        height, width, _ = queries_grid.shape
        wp = self.window_patches
        cp = self.context_patches
        op = self.output_overlap_patches
        assert wp is not None and op > 0
        assert wp + 2 * op <= nw, (
            "An overlapping longitude window must not wrap over itself."
        )

        # With full latent context, adjacent windows make the same prediction
        # for every repeated query. Decode each global query once instead of
        # materializing the complete latent grid once per window.
        if cp is None:
            full_data = rearrange(data_grid, "b nh nw c -> b (nh nw) c")
            all_queries = rearrange(queries_grid, "h w d -> (h w) d")
            decoded = self._decode_queries(full_data, all_queries)
            return rearrange(
                decoded,
                "b (h w) c -> b c h w",
                h=height,
                w=width,
            )

        weighted = data_grid.new_zeros(batch, height * width, self.decoded_channels)
        weight_sum = data_grid.new_zeros(height * width, 1)
        n_blocks_h = nh // wp
        n_blocks_w = nw // wp
        halo_h = op * patch_h
        halo_w = op * patch_w

        data_windows = self._local_data_windows(
            data_grid,
            window_patches=wp,
            context_patches=cp,
        )

        # Polar windows have a shorter latitude query halo than interior
        # windows. Group windows by query height, then split only when the
        # combined model-batch/window dimension exceeds the bounded native
        # kernel batch size.
        latitude_groups: dict[int, list[int]] = {}
        latitude_weights: dict[int, torch.Tensor] = {}
        for bi in range(n_blocks_h):
            query_i0 = max(0, (bi * wp - op) * patch_h)
            query_i1 = min(height, ((bi + 1) * wp + op) * patch_h)
            query_height = query_i1 - query_i0
            latitude_groups.setdefault(query_height, []).append(bi)
            latitude_weights[bi] = self._overlap_weights(
                query_height,
                halo_h,
                fade_start=bi > 0,
                fade_end=bi + 1 < n_blocks_h,
                device=data_grid.device,
                dtype=data_grid.dtype,
            )

        query_width = (wp + 2 * op) * patch_w
        lon_weights = self._overlap_weights(
            query_width,
            halo_w,
            fade_start=True,
            fade_end=True,
            device=data_grid.device,
            dtype=data_grid.dtype,
        )

        for query_height, block_rows in latitude_groups.items():
            block_row_index = torch.tensor(block_rows, device=data_grid.device)
            block_i = block_row_index.repeat_interleave(n_blocks_w)
            block_j = torch.arange(n_blocks_w, device=data_grid.device).repeat(
                len(block_rows)
            )
            window_index = block_i * n_blocks_w + block_j
            lat_start = ((block_i * wp - op) * patch_h).clamp_min(0)
            lat_index = lat_start[:, None] + torch.arange(
                query_height, device=data_grid.device
            )
            lon_start = (block_j * wp - op) * patch_w
            lon_index = (
                lon_start[:, None] + torch.arange(query_width, device=data_grid.device)
            ).remainder(width)
            lat_weight = torch.stack(
                [latitude_weights[bi] for bi in block_rows]
            ).repeat_interleave(n_blocks_w, dim=0)
            weights = lat_weight[:, :, None] * lon_weights[None, None, :]
            queries = queries_grid[
                lat_index[:, :, None],
                lon_index[:, None, :],
            ]
            query_count = query_height * query_width

            local_data = rearrange(
                data_windows.index_select(1, window_index),
                "b n t c -> (b n) t c",
            )
            batched_queries = (
                rearrange(queries, "n h w d -> n (h w) d")
                .unsqueeze(0)
                .expand(batch, -1, -1, -1)
                .reshape(batch * len(window_index), query_count, -1)
            )
            local_out = self._decode_window_batch(local_data, batched_queries)
            local_out = rearrange(
                local_out,
                "(b n) q c -> b (n q) c",
                b=batch,
                n=len(window_index),
            )
            flat_indices = (
                lat_index[:, :, None] * width + lon_index[:, None, :]
            ).flatten()
            flat_weights = weights.flatten()
            weighted.index_add_(
                1,
                flat_indices,
                local_out * flat_weights[None, :, None],
            )
            weight_sum.index_add_(0, flat_indices, flat_weights[:, None])

        if not torch.all(weight_sum > 0):
            raise RuntimeError("Overlapping decoder left output pixels uncovered.")
        return rearrange(
            weighted / weight_sum.unsqueeze(0),
            "b (h w) c -> b c h w",
            h=height,
            w=width,
        )

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
