# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

# Perceiver-based decoder, complementary to encoder.py

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import nn

from samudra.constants import Lat, Lon
from samudra.models.modules.augment_input import make_3d_coordinate_grid
from samudra.models.modules.encoder import patch_from, pos_scale_enc_for_grid


def coordinate_bilinear_resample(
    x: Float[torch.Tensor, "batch channels H_source W_source"],
    source_resolution: tuple[Lat, Lon],
    output_resolution: tuple[Lat, Lon],
    valid_mask: torch.Tensor | None = None,
    *,
    output_latitude_chunk_size: int | None = None,
) -> Float[torch.Tensor, "batch channels H_output W_output"]:
    """Interpolate on physical coordinates with periodic longitude.

    Latitude is edge-clamped. Longitude wraps with a synthetic copy of the first
    source column at ``lon[0] + 360``. Interpolation accumulates in float32 so the
    geometry path remains stable when the surrounding model uses bfloat16. Output
    latitudes are chunked to bound high-resolution corner and mask temporaries.
    """
    source_lat_cpu, source_lon_cpu = source_resolution
    output_lat_cpu, output_lon_cpu = output_resolution
    if x.ndim != 4:
        raise ValueError(f"Expected a four-dimensional feature grid, got {x.ndim}D.")
    if x.shape[-2:] != (len(source_lat_cpu), len(source_lon_cpu)):
        raise ValueError(
            "Source coordinates must match the feature grid; got grid "
            f"{tuple(x.shape[-2:])} and coordinates "
            f"{(len(source_lat_cpu), len(source_lon_cpu))}."
        )
    if len(source_lat_cpu) < 2 or len(source_lon_cpu) < 2:
        raise ValueError("Coordinate resampling requires at least two cells per axis.")
    if (
        x.shape[-2:] == (len(output_lat_cpu), len(output_lon_cpu))
        and torch.equal(source_lat_cpu, output_lat_cpu)
        and torch.equal(source_lon_cpu, output_lon_cpu)
    ):
        return x

    device = x.device
    source_lat = source_lat_cpu.to(device=device, dtype=torch.float32)
    source_lon = source_lon_cpu.to(device=device, dtype=torch.float32)
    output_lat = output_lat_cpu.to(device=device, dtype=torch.float32)
    output_lon = output_lon_cpu.to(device=device, dtype=torch.float32)
    if not torch.all(source_lat[1:] > source_lat[:-1]):
        raise ValueError("Source latitude coordinates must be strictly increasing.")
    if not torch.all(source_lon[1:] > source_lon[:-1]):
        raise ValueError("Source longitude coordinates must be strictly increasing.")
    if source_lon[-1] - source_lon[0] >= 360:
        raise ValueError("Source longitude coordinates must not duplicate the seam.")

    clamped_lat = output_lat.clamp(source_lat[0], source_lat[-1])
    lat_upper = torch.searchsorted(source_lat, clamped_lat, right=True).clamp(
        1, len(source_lat) - 1
    )
    lat_lower = lat_upper - 1
    lat_denominator = source_lat[lat_upper] - source_lat[lat_lower]
    lat_weight = (clamped_lat - source_lat[lat_lower]) / lat_denominator

    wrapped_lon = torch.remainder(output_lon - source_lon[0], 360.0) + source_lon[0]
    extended_lon = torch.cat((source_lon, source_lon[0:1] + 360.0))
    lon_upper = torch.searchsorted(extended_lon, wrapped_lon, right=True).clamp(
        1, len(source_lon)
    )
    lon_lower = lon_upper - 1
    lon_denominator = extended_lon[lon_upper] - extended_lon[lon_lower]
    lon_weight = (wrapped_lon - extended_lon[lon_lower]) / lon_denominator
    lon_upper_wrapped = lon_upper.remainder(len(source_lon))

    mask: torch.Tensor | None = None
    if valid_mask is not None:
        if valid_mask.ndim not in (2, 3) or valid_mask.shape[-2:] != x.shape[-2:]:
            raise ValueError(
                "The resampling validity mask must be [H, W] or [C, H, W] and "
                "match the source grid; got "
                f"{tuple(valid_mask.shape)} and {tuple(x.shape[-2:])}."
            )
        mask = valid_mask.to(device=device, dtype=torch.float32)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.shape[0] not in (1, x.shape[1]):
            raise ValueError(
                "A channel-wise resampling mask must have one mask per feature "
                f"channel; got {mask.shape[0]} masks for {x.shape[1]} channels."
            )
    output_height = len(output_lat)
    if output_latitude_chunk_size is not None and output_latitude_chunk_size < 1:
        raise ValueError("Output latitude chunk size must be positive.")
    if output_latitude_chunk_size is None:
        # Bound each four-corner tensor to roughly 128 MiB in float32. The
        # channel-wise mask weights have comparable size, keeping the main pair
        # near 256 MiB independently of output grid height.
        max_corner_elements = 32 * 1024 * 1024
        elements_per_output_row = 4 * x.shape[0] * x.shape[1] * len(output_lon)
        output_latitude_chunk_size = max(
            1,
            min(output_height, max_corner_elements // elements_per_output_row),
        )

    values = x.to(dtype=torch.float32)
    lower_lon = lon_lower[None, :]
    upper_lon = lon_upper_wrapped[None, :]
    lon_weight_grid = lon_weight[None, :]
    output_chunks: list[torch.Tensor] = []
    for start in range(0, output_height, output_latitude_chunk_size):
        stop = min(start + output_latitude_chunk_size, output_height)
        lower_lat = lat_lower[start:stop, None]
        upper_lat = lat_upper[start:stop, None]
        corners = torch.stack(
            (
                values[:, :, lower_lat, lower_lon],
                values[:, :, lower_lat, upper_lon],
                values[:, :, upper_lat, lower_lon],
                values[:, :, upper_lat, upper_lon],
            ),
            dim=0,
        )
        lat_weight_grid = lat_weight[start:stop, None]
        weights = torch.stack(
            (
                (1 - lat_weight_grid) * (1 - lon_weight_grid),
                (1 - lat_weight_grid) * lon_weight_grid,
                lat_weight_grid * (1 - lon_weight_grid),
                lat_weight_grid * lon_weight_grid,
            ),
            dim=0,
        )
        if mask is not None:
            corner_mask = torch.stack(
                (
                    mask[:, lower_lat, lower_lon],
                    mask[:, lower_lat, upper_lon],
                    mask[:, upper_lat, lower_lon],
                    mask[:, upper_lat, upper_lon],
                ),
                dim=0,
            )
            weights = weights[:, None] * corner_mask
        else:
            weights = weights[:, None]

        weight_sum = weights.sum(dim=0)
        output = (corners * weights[:, None]).sum(dim=0)
        output = output / weight_sum.clamp_min(torch.finfo(torch.float32).eps)[None]
        output_chunks.append(torch.where(weight_sum[None] > 0, output, 0))
    return torch.cat(output_chunks, dim=-2).to(dtype=x.dtype)


class ResampleProjectionDecoder(nn.Module):
    """Resize a canonical feature grid, then project channels per output pixel.

    When the processor and output grids already match, the resize is skipped and
    this is exactly a shared 1-by-1 projection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        coordinate_resampling: bool = False,
        project_before_resample: bool = False,
    ) -> None:
        super().__init__()
        if project_before_resample and not coordinate_resampling:
            raise ValueError(
                "project_before_resample requires coordinate_resampling so the "
                "source validity mask can renormalize interpolation."
            )
        # Callers must supply the per-channel source mask whenever transport
        # actually happens; without it this silently degrades to unmasked
        # interpolation and land values bleed into ocean cells.
        self.requires_source_mask = project_before_resample
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.coordinate_resampling = coordinate_resampling
        self.project_before_resample = project_before_resample
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(
        self,
        x: Float[torch.Tensor, "batch channels H W"],
        resolution: tuple[Lat, Lon],
        *,
        source_resolution: tuple[Lat, Lon] | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> Float[torch.Tensor, "batch {self.out_channels} H_out W_out"]:
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[1]}."
            )
        output_shape = len(resolution[0]), len(resolution[1])
        if self.coordinate_resampling:
            if source_resolution is None:
                raise ValueError(
                    "Coordinate resampling requires the processor-grid resolution."
                )
            if self.project_before_resample:
                x = self.projection(x)
                return coordinate_bilinear_resample(
                    x,
                    source_resolution,
                    resolution,
                    valid_mask=valid_mask,
                )
            x = coordinate_bilinear_resample(x, source_resolution, resolution)
        elif x.shape[-2:] != output_shape:
            x = F.interpolate(
                x,
                size=output_shape,
                mode="bilinear",
                align_corners=False,
            )
        return self.projection(x)


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
        window_patches: int | None,
        context_patches: int | None,
        window_batch_size: int | None = 1,
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
        if window_batch_size is not None and window_batch_size < 1:
            raise ValueError("window_batch_size must be positive or None.")
        self.window_batch_size = window_batch_size

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
        *,
        source_resolution: tuple[Lat, Lon] | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> Float[torch.Tensor, "batch {self.out_channels} H W"]:
        del source_resolution, valid_mask
        # nh, nw: number of patches along height and width (the latent grid dims).
        B, C, nh, nw = x.shape
        lat, lon = resolution

        H, W = len(lat), len(lon)

        pos_patch_h, pos_patch_w = patch_from(self.patch_extent, H, W)

        # --- Add pos/scale encoding to latent tokens (before perceiver, unlike encoder) ---
        tokens = rearrange(x, "b c nh nw -> b (nh nw) c")

        pos_encode, scale_encode = pos_scale_enc_for_grid(
            C,
            lat,
            lon,
            (pos_patch_h, pos_patch_w),
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
        queries_grid: Float[torch.Tensor, "... H W queries_dim"],
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
        H, W = queries_grid.shape[-3:-1]

        if self.window_patches is None:
            data = rearrange(data_grid, "b nh nw c -> b (nh nw) c")
            if queries_grid.ndim == 3:
                queries = rearrange(queries_grid, "h w d -> (h w) d")
            else:
                queries = rearrange(queries_grid, "b h w d -> b (h w) d")
            out = self.perceiver_io(data, queries=queries)  # (B, H*W, out_channels)
            return rearrange(out, "b (h w) c -> b c h w", h=H, w=W)

        wp = self.window_patches
        cp = self.context_patches

        assert nh % wp == 0 and nw % wp == 0, (
            f"Latent grid ({nh}, {nw}) must be divisible by window_patches={wp}"
        )

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

        # --- Batch independent spatial blocks into fewer PerceiverIO calls ---
        n_blocks = n_blocks_h * n_blocks_w
        if cp is None:
            block_data = full_data.unsqueeze(0).expand(n_blocks, -1, -1, -1)
        else:
            block_data = rearrange(
                data_windows,
                "b c bh bw h w -> (bh bw) b (h w) c",
            )
        if queries_grid.ndim == 3:
            block_queries = rearrange(
                queries_grid,
                "(bh h) (bw w) d -> (bh bw) (h w) d",
                bh=n_blocks_h,
                bw=n_blocks_w,
                h=block_ph,
                w=block_pw,
            )
            block_queries = block_queries.unsqueeze(1).expand(-1, B, -1, -1)
        else:
            block_queries = rearrange(
                queries_grid,
                "b (bh h) (bw w) d -> (bh bw) b (h w) d",
                bh=n_blocks_h,
                bw=n_blocks_w,
                h=block_ph,
                w=block_pw,
            )

        blocks_per_call = self.window_batch_size or n_blocks
        decoded_chunks = []
        for start in range(0, n_blocks, blocks_per_call):
            stop = min(start + blocks_per_call, n_blocks)
            local_data = rearrange(
                block_data[start:stop], "blocks b n c -> (blocks b) n c"
            )
            local_queries = rearrange(
                block_queries[start:stop], "blocks b n d -> (blocks b) n d"
            )
            local_out = self.perceiver_io(local_data, queries=local_queries)
            decoded_chunks.append(
                rearrange(
                    local_out,
                    "(blocks b) (h w) c -> blocks b h w c",
                    b=B,
                    h=block_ph,
                    w=block_pw,
                )
            )

        output_blocks = torch.cat(decoded_chunks, dim=0)
        out = rearrange(
            output_blocks,
            "(bh bw) b h w c -> b (bh h) (bw w) c",
            bh=n_blocks_h,
            bw=n_blocks_w,
        )
        return rearrange(out, "b h w c -> b c h w")
