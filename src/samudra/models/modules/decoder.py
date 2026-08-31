# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

# Perceiver-based decoder, complementary to encoder.py

import torch
import torch.nn.functional as F
from aurora.model.fourier import pos_expansion, scale_expansion
from aurora.model.posencoding import pos_scale_enc
from einops import rearrange, repeat
from jaxtyping import Float
from torch import nn

from samudra.constants import Lat, Lon
from samudra.models.modules.augment_input import make_3d_coordinate_grid
from samudra.models.modules.blocks import PointwiseLinear
from samudra.models.modules.encoder import dct_detail_basis, patch_from
from samudra.models.modules.perceiver import (
    Attention,
    AttentionBackend,
    FeedForward,
    PreNorm,
)


def zonally_periodic_bilinear_interpolate(
    x: torch.Tensor,
    size: tuple[int, int],
) -> torch.Tensor:
    """Bilinearly resize a grid while interpolating across the longitude seam."""
    target_height, target_width = size
    width = x.shape[-1]
    if target_width % width:
        raise ValueError(
            f"Target width {target_width} must be an integer multiple of {width}."
        )
    scale_width = target_width // width
    padded = F.pad(x, (1, 1, 0, 0), mode="circular")
    resized = F.interpolate(
        padded,
        size=(target_height, target_width + 2 * scale_width),
        mode="bilinear",
        align_corners=False,
    )
    return resized[..., scale_width : scale_width + target_width]


class StructuredLocalDecoder(nn.Module):
    """Coordinate-resampled base plus a zero-initialized local SDPA residual.

    The base gives every output a smooth, amplitude-preserving route from the
    processor grid.  The residual evaluates query-relative values from a fixed
    neighborhood and blends them with position-anchored attention.  This is the
    compact Samudra implementation of the inverse promoted by Jesse's coarse
    latent/high-resolution dynamics study.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        hidden_dim: int,
        heads: int = 4,
        dim_head: int = 32,
        neighborhood_radius: int = 1,
        position_bias_strength: float = 8.0,
        query_chunk_size: int = 4096,
    ) -> None:
        super().__init__()
        if neighborhood_radius < 0:
            raise ValueError("neighborhood_radius must be non-negative.")
        if query_chunk_size < 1:
            raise ValueError("query_chunk_size must be positive.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dim_head = dim_head
        self.neighborhood_radius = neighborhood_radius
        self.position_bias_strength = position_bias_strength
        self.query_chunk_size = query_chunk_size
        inner_dim = heads * dim_head
        self.base_projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.content_norm = nn.LayerNorm(in_channels)
        self.query_projection = nn.Linear(5, inner_dim, bias=False)
        self.query_hidden_projection = nn.Linear(5, hidden_dim)
        self.key_projection = nn.Linear(in_channels, inner_dim, bias=False)
        self.value_projection = nn.Sequential(
            nn.Linear(in_channels + 2, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, inner_dim),
        )
        self.context_projection = nn.Linear(inner_dim, hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.output_projection = nn.Linear(hidden_dim, out_channels)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def _routing(
        self,
        source_resolution: tuple[Lat, Lon],
        output_resolution: tuple[Lat, Lon],
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        source_lat, source_lon = (
            coordinate.to(device=device, dtype=torch.float32)
            for coordinate in source_resolution
        )
        output_lat, output_lon = (
            coordinate.to(device=device, dtype=torch.float32)
            for coordinate in output_resolution
        )
        lat_spacing = (source_lat[1:] - source_lat[:-1]).median()
        lon_spacing = (source_lon[1:] - source_lon[:-1]).median()
        lat_position = (output_lat - (source_lat[0] - lat_spacing / 2)) / lat_spacing
        lon_position = (
            torch.remainder(output_lon - (source_lon[0] - lon_spacing / 2), 360.0)
            / lon_spacing
        )
        lat_index = lat_position.floor().long().clamp(0, len(source_lat) - 1)
        lon_index = lon_position.floor().long().remainder(len(source_lon))
        relative_lat = 2 * (lat_position - lat_index) - 1
        relative_lon = 2 * (lon_position - lon_index) - 1
        lat_grid, lon_grid = torch.meshgrid(lat_index, lon_index, indexing="ij")
        relative_lat_grid, relative_lon_grid = torch.meshgrid(
            relative_lat, relative_lon, indexing="ij"
        )
        offsets = torch.arange(
            -self.neighborhood_radius,
            self.neighborhood_radius + 1,
            device=device,
        )
        offset_lat, offset_lon = torch.meshgrid(offsets, offsets, indexing="ij")
        offset_lat = offset_lat.flatten()
        offset_lon = offset_lon.flatten()
        neighbor_lat = (lat_grid[..., None] + offset_lat).clamp(0, len(source_lat) - 1)
        neighbor_lon = (lon_grid[..., None] + offset_lon).remainder(len(source_lon))
        neighbor_indices = (neighbor_lat * len(source_lon) + neighbor_lon).flatten(0, 1)
        relative_to_neighbor = torch.stack(
            (
                relative_lat_grid.flatten()[:, None] / 2 - offset_lat,
                relative_lon_grid.flatten()[:, None] / 2 - offset_lon,
            ),
            dim=-1,
        )
        position_bias = (
            -self.position_bias_strength * relative_to_neighbor.square().sum(dim=-1)
        )
        output_lat_grid, output_lon_grid = torch.meshgrid(
            output_lat, output_lon, indexing="ij"
        )
        lat_radians = torch.deg2rad(output_lat_grid)
        lon_radians = torch.deg2rad(output_lon_grid)
        query_features = torch.stack(
            (
                torch.cos(lat_radians) * torch.cos(lon_radians),
                torch.cos(lat_radians) * torch.sin(lon_radians),
                torch.sin(lat_radians),
                torch.full_like(
                    output_lat_grid,
                    float((180.0 / len(output_lat) / lat_spacing).item()),
                ),
                torch.full_like(
                    output_lon_grid,
                    float((360.0 / len(output_lon) / lon_spacing).item()),
                ),
            ),
            dim=-1,
        ).flatten(0, 1)
        return neighbor_indices, position_bias, relative_to_neighbor, query_features

    def forward(
        self,
        x: Float[torch.Tensor, "batch channels H_source W_source"],
        resolution: tuple[Lat, Lon],
        *,
        source_resolution: tuple[Lat, Lon],
    ) -> Float[torch.Tensor, "batch channels_out H_output W_output"]:
        batch, channels, source_height, source_width = x.shape
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {channels}.")
        if (source_height, source_width) != tuple(map(len, source_resolution)):
            raise ValueError("Source coordinates must match the processor grid.")
        output_shape = (len(resolution[0]), len(resolution[1]))
        base = self.base_projection(
            zonally_periodic_bilinear_interpolate(x, output_shape)
        )
        neighbor_indices, position_bias, relative_offsets, query_features = (
            self._routing(source_resolution, resolution, device=x.device)
        )
        content = rearrange(x, "b c h w -> b (h w) c")
        keys = self.key_projection(self.content_norm(content))
        keys = rearrange(keys, "b n (h d) -> b h n d", h=self.heads)
        chunks: list[torch.Tensor] = []
        for start in range(0, len(query_features), self.query_chunk_size):
            stop = min(start + self.query_chunk_size, len(query_features))
            indices = neighbor_indices[start:stop]
            query_count, neighbors = indices.shape
            local_content = content[:, indices]
            offsets = relative_offsets[start:stop].to(dtype=x.dtype)
            offsets = offsets[None].expand(batch, -1, -1, -1)
            values = self.value_projection(torch.cat((local_content, offsets), dim=-1))
            values = rearrange(values, "b q k (h d) -> (b q) h k d", h=self.heads)
            local_keys = keys[:, :, indices]
            local_keys = rearrange(local_keys, "b h q k d -> (b q) h k d")
            query = self.query_projection(query_features[start:stop].to(dtype=x.dtype))
            query = rearrange(query, "q (h d) -> q h 1 d", h=self.heads)
            query = repeat(query, "q h one d -> (b q) h one d", b=batch)
            bias = position_bias[start:stop].to(dtype=x.dtype)
            bias = repeat(bias, "q k -> (b q) h 1 k", b=batch, h=self.heads)
            context = F.scaled_dot_product_attention(
                query, local_keys, values, attn_mask=bias
            )
            context = rearrange(
                context, "(b q) h 1 d -> b q (h d)", b=batch, q=query_count
            )
            query_chunk = query_features[start:stop].to(dtype=x.dtype)
            hidden = self.context_projection(context)
            hidden = hidden + self.query_hidden_projection(query_chunk)[None]
            hidden = hidden + self.feed_forward(hidden)
            chunks.append(self.output_projection(hidden))
        correction = torch.cat(chunks, dim=1)
        correction = rearrange(
            correction,
            "b (h w) c -> b c h w",
            h=output_shape[0],
            w=output_shape[1],
        )
        return base + correction


class PixelRefinementBlock(nn.Module):
    """Identity-initialized full-resolution refinement with spherical padding."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=0,
            groups=channels,
        )
        self.pointwise = PointwiseLinear(channels, channels)
        nn.init.zeros_(self.pointwise.linear.weight)
        nn.init.zeros_(self.pointwise.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x.movedim(1, -1)).movedim(-1, 1)
        x = F.pad(x, (1, 1, 0, 0), mode="circular")
        x = F.pad(x, (0, 0, 1, 1), mode="replicate")
        x = F.gelu(self.depthwise(x))
        return residual + self.pointwise(x)


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


class DCTDetailDecoder(nn.Module):
    """Synthesize output patches from learned coefficients in a fixed DCT basis."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_extent: tuple[float, float],
        *,
        detail_count: int,
        pixel_refinement: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_extent = patch_extent
        self.detail_count = detail_count
        self.coefficient_projection = nn.Conv2d(
            in_channels,
            out_channels * (detail_count + 1),
            kernel_size=1,
        )
        self.pixel_refiner = (
            PixelRefinementBlock(out_channels) if pixel_refinement else None
        )

    def forward(
        self,
        x: Float[torch.Tensor, "batch channels nh nw"],
        resolution: tuple[Lat, Lon],
    ) -> Float[torch.Tensor, "batch {self.out_channels} H W"]:
        batch, channels, coarse_h, coarse_w = x.shape
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {channels}.")
        lat, lon = resolution
        height, width = len(lat), len(lon)
        patch_h, patch_w = patch_from(self.patch_extent, height, width)
        if (coarse_h * patch_h, coarse_w * patch_w) != (height, width):
            raise ValueError(
                "DCT decoder latent grid and patch extent do not cover the output "
                f"grid: latent={(coarse_h, coarse_w)}, patch={(patch_h, patch_w)}, "
                f"output={(height, width)}."
            )
        coefficients = self.coefficient_projection(x)
        coefficients = rearrange(
            coefficients,
            "b (c k) h w -> b h w c k",
            c=self.out_channels,
            k=self.detail_count + 1,
        )
        basis = dct_detail_basis(
            patch_h,
            patch_w,
            self.detail_count,
            device=x.device,
            dtype=x.dtype,
        )
        patches = torch.einsum("bhwck,nk->bhwcn", coefficients, basis)
        patches = rearrange(
            patches,
            "b h w c (ph pw) -> b c (h ph) (w pw)",
            ph=patch_h,
            pw=patch_w,
        )
        if self.pixel_refiner is not None:
            patches = self.pixel_refiner(patches)
        return patches


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
        processor_conditioning: bool = False,
        pixel_refinement: bool = False,
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
        self.decode_hidden_features = processor_conditioning or pixel_refinement
        if self.decode_hidden_features and not isinstance(
            perceiver_io, DirectCrossAttentionIO
        ):
            raise ValueError(
                "processor conditioning and pixel refinement currently require "
                "the direct_cross_attention decoder architecture."
            )

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
        self.pixel_refiner = (
            PixelRefinementBlock(queries_dim) if pixel_refinement else None
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

        if self.decode_hidden_features:
            if self.processor_conditioner is not None:
                processor = zonally_periodic_bilinear_interpolate(x, (H, W))
                conditioning = self.processor_conditioner(processor)
                assert self.conditioning_strength is not None
                out = out + self.conditioning_strength * conditioning
            if self.pixel_refiner is not None:
                out = self.pixel_refiner(out)
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
            if self.decode_hidden_features
            else self.out_channels
        )

    def _decode_queries(
        self,
        data: torch.Tensor,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        if not self.decode_hidden_features:
            return self.perceiver_io(data, queries=queries)
        direct_decoder = self.perceiver_io
        assert isinstance(direct_decoder, DirectCrossAttentionIO)
        return direct_decoder.decode_features(data, queries=queries)

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
        out = data_grid.new_zeros(B, H, W, self.decoded_channels)

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

                local_out = self._decode_queries(local_data, local_queries)
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

        weighted = data_grid.new_zeros(B, H, W, self.decoded_channels)
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
                local_out = self._decode_queries(local_data, local_queries)
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
