# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Probe learned restriction and prolongation through a fixed coarse latent grid.

The synthetic physical grids are integer refinements of one canonical coarse
grid. Unlike ``probe_perceiver_decoder.py``, every decoder input token summarizes
multiple physical cells. This makes the probe relevant to SamudraMulti's
3-degree-by-5-degree patch representation.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from samudra.config import PerceiverConfig
from samudra.models.modules.decoder import ResampleProjectionDecoder
from samudra.models.modules.encoder import PerceiverEncoder

EncoderName = Literal["mean", "perceiver", "attention", "resolved", "moments"]
DecoderName = Literal["bilinear", "coordinate", "anchored", "hybrid"]


@dataclass(frozen=True)
class ProbeConfig:
    encoder: EncoderName
    decoder: DecoderName
    coarse_height: int
    coarse_width: int
    input_channels: int
    latent_channels: int
    batch_size: int
    eval_samples: int
    steps: int
    learning_rate: float
    attention_heads: int
    attention_dim_head: int
    moment_count: int
    neighborhood_radius: int
    position_bias_strength: float
    spectral_decay: float
    seed: int
    device: str


ROUTES = {
    "3x5_to_3x5": ((3, 5), (3, 5)),
    "3x5_to_6x10": ((3, 5), (6, 10)),
    "6x10_to_3x5": ((6, 10), (3, 5)),
    "6x10_to_6x10": ((6, 10), (6, 10)),
}

FOURIER_MODES = (
    (0, 0),
    (1, 0),
    (0, 1),
    (1, 1),
    (2, 1),
    (1, 2),
    (3, 2),
    (2, 4),
    (5, 3),
    (4, 7),
    (7, 5),
    (9, 8),
    (11, 10),
    (14, 13),
)


def parse_args() -> ProbeConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--encoder",
        choices=("mean", "perceiver", "attention", "resolved", "moments"),
        default="resolved",
    )
    parser.add_argument(
        "--decoder",
        choices=("bilinear", "coordinate", "anchored", "hybrid"),
        default="anchored",
    )
    parser.add_argument("--coarse-height", type=int, default=12)
    parser.add_argument("--coarse-width", type=int, default=12)
    parser.add_argument("--input-channels", type=int, default=8)
    parser.add_argument("--latent-channels", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--attention-dim-head", type=int, default=32)
    parser.add_argument("--moment-count", type=int, default=12)
    parser.add_argument("--neighborhood-radius", type=int, default=1)
    parser.add_argument("--position-bias-strength", type=float, default=8.0)
    parser.add_argument(
        "--spectral-decay",
        type=float,
        default=1.0,
        help="Power applied to mode wavenumber attenuation; zero gives equal energy.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device, or 'auto' to select CUDA when available.",
    )
    args = parser.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.coarse_height < 2 or args.coarse_width < 2:
        parser.error("Coarse grid dimensions must both be at least two.")
    if args.latent_channels < args.input_channels:
        parser.error("Latent width must be at least the input channel count.")
    if args.latent_channels % args.attention_heads:
        parser.error("Latent width must be divisible by the attention head count.")
    if args.spectral_decay < 0:
        parser.error("Spectral decay must be non-negative.")
    if args.moment_count < 1:
        parser.error("Moment count must be positive.")
    return ProbeConfig(**vars(args))


def make_resolution(
    height: int, width: int, *, longitude_shift: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    spacing_lat = 180.0 / height
    spacing_lon = 360.0 / width
    latitude = torch.linspace(
        -90.0 + spacing_lat / 2,
        90.0 - spacing_lat / 2,
        height,
    )
    longitude = (
        torch.arange(width, dtype=torch.float32) + 0.5 + longitude_shift
    ) * spacing_lon
    return latitude, longitude


def coarse_resolution(
    resolution: tuple[torch.Tensor, torch.Tensor],
    coarse_height: int,
    coarse_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    latitude, longitude = resolution
    if len(latitude) % coarse_height or len(longitude) % coarse_width:
        raise ValueError("Physical grid must divide exactly into the coarse grid.")
    return (
        rearrange(latitude, "(h ph) -> h ph", h=coarse_height).mean(dim=-1),
        rearrange(longitude, "(w pw) -> w pw", w=coarse_width).mean(dim=-1),
    )


def analytic_fields(
    coefficients: torch.Tensor,
    resolution: tuple[torch.Tensor, torch.Tensor],
    *,
    spectral_decay: float = 1.0,
) -> torch.Tensor:
    """Evaluate multiscale continuous fields from shared random coefficients."""

    latitude, longitude = resolution
    y = torch.deg2rad(latitude)[:, None]
    x = torch.deg2rad(longitude)[None, :]
    basis: list[torch.Tensor] = []
    for mode_index, (ky, kx) in enumerate(FOURIER_MODES):
        if ky == 0 and kx == 0:
            value = torch.ones(len(latitude), len(longitude))
        else:
            phase = math.pi / 7 * mode_index
            value = torch.sin(ky * y + kx * x + phase)
            value = value + 0.5 * torch.cos(kx * x - ky * y + 0.5 * phase)
        basis.append(value)
    stacked = torch.stack(basis).to(
        device=coefficients.device, dtype=coefficients.dtype
    )
    mode_scale = torch.tensor(
        [
            (1 + ky * ky + kx * kx) ** (-0.5 * spectral_decay)
            for ky, kx in FOURIER_MODES
        ],
        device=coefficients.device,
        dtype=coefficients.dtype,
    )
    stacked = stacked * mode_scale[:, None, None]
    return torch.einsum("bcm,mhw->bchw", coefficients, stacked)


def make_coefficients(
    *,
    samples: int,
    channels: int,
    seed: int,
    device: str,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(
        samples,
        channels,
        len(FOURIER_MODES),
        generator=generator,
    ).to(device)


def patch_tokens(
    x: torch.Tensor, coarse_height: int, coarse_width: int
) -> tuple[torch.Tensor, int, int]:
    _, _, height, width = x.shape
    if height % coarse_height or width % coarse_width:
        raise ValueError("Input grid must divide exactly into the coarse grid.")
    patch_height = height // coarse_height
    patch_width = width // coarse_width
    tokens = rearrange(
        x,
        "b c (h ph) (w pw) -> (b h w) (ph pw) c",
        h=coarse_height,
        w=coarse_width,
    )
    return tokens, patch_height, patch_width


def relative_patch_coordinates(
    patch_height: int,
    patch_width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    y = torch.arange(patch_height, device=device, dtype=dtype) + 0.5
    x = torch.arange(patch_width, device=device, dtype=dtype) + 0.5
    y = 2 * y / patch_height - 1
    x = 2 * x / patch_width - 1
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((yy, xx), dim=-1).flatten(0, 1)


class PatchAttentionPool(nn.Module):
    """Pool variable-size physical patches while preserving raw value amplitudes."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        heads: int,
    ) -> None:
        super().__init__()
        if out_channels % heads:
            raise ValueError("Attention-pool width must be divisible by heads.")
        self.out_channels = out_channels
        self.heads = heads
        self.dim_head = out_channels // heads
        self.content_norm = nn.LayerNorm(in_channels)
        self.key = nn.Linear(in_channels + 2, out_channels)
        self.value = nn.Linear(in_channels, out_channels)
        self.query = nn.Parameter(torch.randn(heads, self.dim_head) * 0.02)
        self.output = nn.Linear(out_channels, out_channels)

    def forward(self, tokens: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        normalized = self.content_norm(tokens)
        coords = coordinates[None].expand(tokens.shape[0], -1, -1)
        keys = self.key(torch.cat((normalized, coords), dim=-1))
        values = self.value(tokens)
        keys = rearrange(keys, "b n (h d) -> b h n d", h=self.heads)
        values = rearrange(values, "b n (h d) -> b h n d", h=self.heads)
        logits = torch.einsum("hd,bhnd->bhn", self.query, keys)
        logits = logits / math.sqrt(self.dim_head)
        pooled = torch.einsum("bhn,bhnd->bhd", logits.softmax(dim=-1), values)
        return self.output(rearrange(pooled, "b h d -> b (h d)"))


class AttentionPatchEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        coarse_height: int,
        coarse_width: int,
        *,
        heads: int,
    ) -> None:
        super().__init__()
        self.coarse_height = coarse_height
        self.coarse_width = coarse_width
        self.pool = PatchAttentionPool(in_channels, latent_channels, heads=heads)

    def forward(
        self, x: torch.Tensor, resolution: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        del resolution
        batch = x.shape[0]
        tokens, patch_height, patch_width = patch_tokens(
            x, self.coarse_height, self.coarse_width
        )
        coordinates = relative_patch_coordinates(
            patch_height,
            patch_width,
            device=x.device,
            dtype=x.dtype,
        )
        pooled = self.pool(tokens, coordinates)
        return rearrange(
            pooled,
            "(b h w) c -> b c h w",
            b=batch,
            h=self.coarse_height,
            w=self.coarse_width,
        )


class MeanPatchEncoder(nn.Module):
    """Patch-mean-only control with no access to subpatch anomalies."""

    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        coarse_height: int,
        coarse_width: int,
    ) -> None:
        super().__init__()
        self.coarse_height = coarse_height
        self.coarse_width = coarse_width
        self.projection = nn.Linear(in_channels, latent_channels)

    def forward(
        self, x: torch.Tensor, resolution: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        del resolution
        batch = x.shape[0]
        tokens, _, _ = patch_tokens(x, self.coarse_height, self.coarse_width)
        latent = self.projection(tokens.mean(dim=1))
        return rearrange(
            latent,
            "(b h w) c -> b c h w",
            b=batch,
            h=self.coarse_height,
            w=self.coarse_width,
        )


class ResolvedSubgridEncoder(nn.Module):
    """Concatenate a patch-mean route with learned anomaly attention pooling."""

    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        coarse_height: int,
        coarse_width: int,
        *,
        heads: int,
    ) -> None:
        super().__init__()
        self.coarse_height = coarse_height
        self.coarse_width = coarse_width
        mean_channels = max(in_channels, latent_channels // 4)
        mean_channels = min(mean_channels, latent_channels - heads)
        subgrid_channels = latent_channels - mean_channels
        subgrid_channels -= subgrid_channels % heads
        mean_channels = latent_channels - subgrid_channels
        self.mean_projection = nn.Linear(in_channels, mean_channels)
        self.subgrid_pool = PatchAttentionPool(
            in_channels, subgrid_channels, heads=heads
        )

    def forward(
        self, x: torch.Tensor, resolution: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        del resolution
        batch = x.shape[0]
        tokens, patch_height, patch_width = patch_tokens(
            x, self.coarse_height, self.coarse_width
        )
        mean = tokens.mean(dim=1)
        anomaly = tokens - mean[:, None]
        coordinates = relative_patch_coordinates(
            patch_height,
            patch_width,
            device=x.device,
            dtype=x.dtype,
        )
        latent = torch.cat(
            (
                self.mean_projection(mean),
                self.subgrid_pool(anomaly, coordinates),
            ),
            dim=-1,
        )
        return rearrange(
            latent,
            "(b h w) c -> b c h w",
            b=batch,
            h=self.coarse_height,
            w=self.coarse_width,
        )


class ResolvedMomentEncoder(nn.Module):
    """Pack coarse means and multiple relative-coordinate moments into channels."""

    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        coarse_height: int,
        coarse_width: int,
        *,
        moment_count: int,
    ) -> None:
        super().__init__()
        self.coarse_height = coarse_height
        self.coarse_width = coarse_width
        self.moment_count = moment_count
        mean_channels = max(in_channels, latent_channels // 4)
        mean_channels = min(mean_channels, latent_channels - 1)
        subgrid_channels = latent_channels - mean_channels
        self.mean_projection = nn.Linear(in_channels, mean_channels)
        self.coordinate_basis = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
            nn.Linear(64, moment_count),
        )
        self.moment_projection = nn.Linear(in_channels * moment_count, subgrid_channels)

    def forward(
        self, x: torch.Tensor, resolution: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        del resolution
        batch = x.shape[0]
        tokens, patch_height, patch_width = patch_tokens(
            x, self.coarse_height, self.coarse_width
        )
        mean = tokens.mean(dim=1)
        anomaly = tokens - mean[:, None]
        coordinates = relative_patch_coordinates(
            patch_height,
            patch_width,
            device=x.device,
            dtype=x.dtype,
        )
        basis = self.coordinate_basis(coordinates)
        basis = basis - basis.mean(dim=0, keepdim=True)
        basis = basis / basis.square().mean(dim=0, keepdim=True).sqrt().clamp_min(1e-6)
        moments = torch.einsum("bnc,nm->bcm", anomaly, basis) / tokens.shape[1]
        latent = torch.cat(
            (
                self.mean_projection(mean),
                self.moment_projection(moments.flatten(1)),
            ),
            dim=-1,
        )
        return rearrange(
            latent,
            "(b h w) c -> b c h w",
            b=batch,
            h=self.coarse_height,
            w=self.coarse_width,
        )


class CoordinateConditionedDecoder(nn.Module):
    """Decode each output coordinate from its containing coarse feature token."""

    def __init__(self, in_channels: int, out_channels: int, hidden: int) -> None:
        super().__init__()
        coordinate_channels = 7
        self.network = nn.Sequential(
            nn.Conv2d(in_channels + coordinate_channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, out_channels, kernel_size=1),
        )

    @staticmethod
    def _indices_and_features(
        source_resolution: tuple[torch.Tensor, torch.Tensor],
        output_resolution: tuple[torch.Tensor, torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        lat_origin = source_lat[0] - lat_spacing / 2
        lon_origin = source_lon[0] - lon_spacing / 2
        lat_position = (output_lat - lat_origin) / lat_spacing
        lon_position = torch.remainder(output_lon - lon_origin, 360.0) / lon_spacing
        lat_index = lat_position.floor().long().clamp(0, len(source_lat) - 1)
        lon_index = lon_position.floor().long().remainder(len(source_lon))
        relative_lat = 2 * (lat_position - lat_index) - 1
        relative_lon = 2 * (lon_position - lon_index) - 1

        latitude_grid, longitude_grid = torch.meshgrid(
            output_lat, output_lon, indexing="ij"
        )
        relative_lat_grid, relative_lon_grid = torch.meshgrid(
            relative_lat, relative_lon, indexing="ij"
        )
        lat_indices, lon_indices = torch.meshgrid(lat_index, lon_index, indexing="ij")
        latitude_radians = torch.deg2rad(latitude_grid)
        longitude_radians = torch.deg2rad(longitude_grid)
        output_lat_spacing = 180.0 / len(output_lat)
        output_lon_spacing = 360.0 / len(output_lon)
        features = torch.stack(
            (
                relative_lat_grid,
                relative_lon_grid,
                torch.cos(latitude_radians) * torch.cos(longitude_radians),
                torch.cos(latitude_radians) * torch.sin(longitude_radians),
                torch.sin(latitude_radians),
                torch.full_like(
                    latitude_grid,
                    float((output_lat_spacing / lat_spacing).item()),
                ),
                torch.full_like(
                    longitude_grid,
                    float((output_lon_spacing / lon_spacing).item()),
                ),
            )
        ).to(dtype=dtype)
        return lat_indices, lon_indices, features

    def forward(
        self,
        x: torch.Tensor,
        resolution: tuple[torch.Tensor, torch.Tensor],
        *,
        source_resolution: tuple[torch.Tensor, torch.Tensor] | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del valid_mask
        if source_resolution is None:
            raise ValueError("Coordinate decoder requires source coordinates.")
        lat_indices, lon_indices, features = self._indices_and_features(
            source_resolution,
            resolution,
            device=x.device,
            dtype=x.dtype,
        )
        local = x[:, :, lat_indices, lon_indices]
        features = features.to(device=x.device)[None].expand(x.shape[0], -1, -1, -1)
        return self.network(torch.cat((local, features), dim=1))


class AnchoredQueryDecoder(nn.Module):
    """Local direct cross-attention with an explicit output-query residual."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        hidden: int,
        heads: int,
        dim_head: int,
        neighborhood_radius: int,
        position_bias_strength: float,
        zero_initialize_output: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.heads = heads
        self.dim_head = dim_head
        self.neighborhood_radius = neighborhood_radius
        self.position_bias_strength = position_bias_strength
        inner_dim = heads * dim_head
        self.scale = dim_head**-0.5
        self.content_norm = nn.LayerNorm(in_channels)
        # Absolute unit-sphere coordinates and output/source scale ratios are
        # continuous across coarse-patch boundaries. Relative coordinates are
        # applied per neighboring token below instead of entering this residual.
        self.query_key_projection = nn.Linear(5, inner_dim, bias=False)
        self.query_hidden_projection = nn.Linear(5, hidden)
        self.key_projection = nn.Linear(in_channels, inner_dim, bias=False)
        self.value_projection = nn.Sequential(
            nn.Linear(in_channels + 2, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, inner_dim),
        )
        self.context_projection = nn.Linear(inner_dim, hidden)
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden),
        )
        self.output_projection = nn.Linear(hidden, out_channels)
        if zero_initialize_output:
            nn.init.zeros_(self.output_projection.weight)
            nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        x: torch.Tensor,
        resolution: tuple[torch.Tensor, torch.Tensor],
        *,
        source_resolution: tuple[torch.Tensor, torch.Tensor] | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del valid_mask
        if source_resolution is None:
            raise ValueError("Anchored decoder requires source coordinates.")
        lat_indices, lon_indices, query_features = (
            CoordinateConditionedDecoder._indices_and_features(
                source_resolution,
                resolution,
                device=x.device,
                dtype=x.dtype,
            )
        )
        output_height, output_width = lat_indices.shape
        offsets = torch.arange(
            -self.neighborhood_radius,
            self.neighborhood_radius + 1,
            device=x.device,
        )
        offset_latitude, offset_longitude = torch.meshgrid(
            offsets, offsets, indexing="ij"
        )
        offset_latitude = offset_latitude.flatten()
        offset_longitude = offset_longitude.flatten()
        neighbor_latitude = (lat_indices[..., None] + offset_latitude).clamp(
            0, x.shape[-2] - 1
        )
        neighbor_longitude = (lon_indices[..., None] + offset_longitude).remainder(
            x.shape[-1]
        )
        flat_indices = (neighbor_latitude * x.shape[-1] + neighbor_longitude).flatten(
            0, 1
        )

        relative_latitude = query_features[0].flatten() / 2
        relative_longitude = query_features[1].flatten() / 2
        distance_squared = (relative_latitude[:, None] - offset_latitude).square() + (
            relative_longitude[:, None] - offset_longitude
        ).square()
        position_bias = -self.position_bias_strength * distance_squared

        content = rearrange(x, "b c h w -> b (h w) c")
        keys = self.key_projection(self.content_norm(content))
        keys = rearrange(keys, "b n (h d) -> b h n d", h=self.heads)
        local_keys = keys[:, :, flat_indices]
        query_features = rearrange(query_features, "c h w -> (h w) c")
        query_continuous = query_features[:, 2:]
        queries = self.query_key_projection(query_continuous)
        queries = rearrange(queries, "q (h d) -> h q d", h=self.heads)
        local_content = content[:, flat_indices]
        relative_to_neighbor = torch.stack(
            (
                relative_latitude[:, None] - offset_latitude,
                relative_longitude[:, None] - offset_longitude,
            ),
            dim=-1,
        )
        relative_to_neighbor = relative_to_neighbor[None].expand(x.shape[0], -1, -1, -1)
        local_values = self.value_projection(
            torch.cat((local_content, relative_to_neighbor), dim=-1)
        )
        local_values = rearrange(local_values, "b q k (h d) -> b h q k d", h=self.heads)
        logits = torch.einsum("hqd,bhqkd->bhqk", queries, local_keys) * self.scale
        attention = (logits + position_bias[None, None]).softmax(dim=-1)
        context = torch.einsum("bhqk,bhqkd->bhqd", attention, local_values)
        context = rearrange(context, "b h q d -> b q (h d)")
        hidden = self.context_projection(context)
        hidden = hidden + self.query_hidden_projection(query_continuous)[None]
        hidden = hidden + self.feed_forward(hidden)
        output = self.output_projection(hidden)
        return rearrange(
            output,
            "b (h w) c -> b c h w",
            h=output_height,
            w=output_width,
        )


class QueryResidualHybridDecoder(nn.Module):
    def __init__(
        self,
        base: ResampleProjectionDecoder,
        correction: AnchoredQueryDecoder,
    ) -> None:
        super().__init__()
        self.base = base
        self.correction = correction

    def forward(
        self,
        x: torch.Tensor,
        resolution: tuple[torch.Tensor, torch.Tensor],
        *,
        source_resolution: tuple[torch.Tensor, torch.Tensor] | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base = self.base(
            x,
            resolution,
            source_resolution=source_resolution,
            valid_mask=valid_mask,
        )
        correction = self.correction(
            x,
            resolution,
            source_resolution=source_resolution,
            valid_mask=valid_mask,
        )
        return base + correction


def build_encoder(config: ProbeConfig) -> nn.Module:
    patch_extent = (
        180.0 / config.coarse_height,
        360.0 / config.coarse_width,
    )
    if config.encoder == "mean":
        return MeanPatchEncoder(
            config.input_channels,
            config.latent_channels,
            config.coarse_height,
            config.coarse_width,
        )
    if config.encoder == "perceiver":
        perceiver_config = PerceiverConfig(
            depth=2,
            latent_dim=64,
            num_latents=64,
            normalize_input_context=True,
            normalize_encoder_output=True,
        )
        perceiver = perceiver_config.build(
            config.input_channels,
            config.latent_channels,
            max_patch_size=(6, 10),
            implementation="naive",
        )
        return PerceiverEncoder(
            config.input_channels,
            config.latent_channels,
            patch_extent,
            perceiver,
            geometry_mode="none",
        )
    if config.encoder == "attention":
        return AttentionPatchEncoder(
            config.input_channels,
            config.latent_channels,
            config.coarse_height,
            config.coarse_width,
            heads=config.attention_heads,
        )
    if config.encoder == "resolved":
        return ResolvedSubgridEncoder(
            config.input_channels,
            config.latent_channels,
            config.coarse_height,
            config.coarse_width,
            heads=config.attention_heads,
        )
    return ResolvedMomentEncoder(
        config.input_channels,
        config.latent_channels,
        config.coarse_height,
        config.coarse_width,
        moment_count=config.moment_count,
    )


def build_decoder(config: ProbeConfig) -> nn.Module:
    if config.decoder == "bilinear":
        return ResampleProjectionDecoder(
            config.latent_channels,
            config.input_channels,
            coordinate_resampling=True,
        )
    if config.decoder == "coordinate":
        return CoordinateConditionedDecoder(
            config.latent_channels,
            config.input_channels,
            hidden=config.latent_channels,
        )
    if config.decoder == "anchored":
        return AnchoredQueryDecoder(
            config.latent_channels,
            config.input_channels,
            hidden=config.latent_channels,
            heads=config.attention_heads,
            dim_head=config.attention_dim_head,
            neighborhood_radius=config.neighborhood_radius,
            position_bias_strength=config.position_bias_strength,
        )
    base = ResampleProjectionDecoder(
        config.latent_channels,
        config.input_channels,
        coordinate_resampling=True,
    )
    correction = AnchoredQueryDecoder(
        config.latent_channels,
        config.input_channels,
        hidden=config.latent_channels,
        heads=config.attention_heads,
        dim_head=config.attention_dim_head,
        neighborhood_radius=config.neighborhood_radius,
        position_bias_strength=config.position_bias_strength,
        zero_initialize_output=True,
    )
    return QueryResidualHybridDecoder(base, correction)


class CoarseLatentProbe(nn.Module):
    def __init__(self, config: ProbeConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = build_encoder(config)
        self.decoder = build_decoder(config)

    def forward(
        self,
        source: torch.Tensor,
        source_resolution: tuple[torch.Tensor, torch.Tensor],
        output_resolution: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(source, source_resolution)
        latent_resolution = coarse_resolution(
            source_resolution,
            self.config.coarse_height,
            self.config.coarse_width,
        )
        output = self.decoder(
            latent,
            output_resolution,
            source_resolution=latent_resolution,
        )
        return output, latent


def spectral_ratios(
    output: torch.Tensor,
    target: torch.Tensor,
    *,
    coarse_height: int,
    coarse_width: int,
) -> dict[str, float]:
    output_power = torch.fft.rfft2(output.float(), norm="ortho").abs().square()
    target_power = torch.fft.rfft2(target.float(), norm="ortho").abs().square()
    latitude_frequency = torch.fft.fftfreq(output.shape[-2], device=output.device)
    longitude_frequency = torch.fft.rfftfreq(output.shape[-1], device=output.device)
    radius = torch.sqrt(
        latitude_frequency[:, None].square() + longitude_frequency[None, :].square()
    )
    high = radius >= 0.25
    latitude_mode = (
        torch.fft.fftfreq(output.shape[-2], device=output.device) * output.shape[-2]
    )
    longitude_mode = (
        torch.fft.rfftfreq(output.shape[-1], device=output.device) * output.shape[-1]
    )
    above_coarse_nyquist = (latitude_mode[:, None].abs() > coarse_height / 2) | (
        longitude_mode[None, :] > coarse_width / 2
    )

    def ratio(mask: torch.Tensor) -> float:
        return float(
            (
                output_power[..., mask].mean()
                / target_power[..., mask].mean().clamp_min(1e-12)
            ).cpu()
        )

    return {
        "high_wavenumber_power_ratio": ratio(high),
        "subpatch_power_ratio": ratio(above_coarse_nyquist),
    }


def seam_metrics(
    output: torch.Tensor,
    target: torch.Tensor,
    patch_height: int,
    patch_width: int,
) -> dict[str, float]:
    latitude_error = (
        (output[..., 1:, :] - output[..., :-1, :])
        - (target[..., 1:, :] - target[..., :-1, :])
    ).abs()
    longitude_error = (
        (output[..., :, 1:] - output[..., :, :-1])
        - (target[..., :, 1:] - target[..., :, :-1])
    ).abs()
    lat_seams = torch.arange(
        patch_height - 1,
        latitude_error.shape[-2],
        patch_height,
        device=output.device,
    )
    lon_seams = torch.arange(
        patch_width - 1,
        longitude_error.shape[-1],
        patch_width,
        device=output.device,
    )
    seam_error = torch.cat(
        (
            latitude_error.index_select(-2, lat_seams).flatten(),
            longitude_error.index_select(-1, lon_seams).flatten(),
        )
    ).mean()
    all_error = torch.cat((latitude_error.flatten(), longitude_error.flatten())).mean()
    return {
        "seam_gradient_mae": float(seam_error.cpu()),
        "seam_to_all_gradient_error": float(
            (seam_error / all_error.clamp_min(1e-12)).cpu()
        ),
    }


def route_tensors(
    config: ProbeConfig,
    route: tuple[tuple[int, int], tuple[int, int]],
    *,
    coefficients: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]:
    source_patch, output_patch = route
    source_shape = (
        config.coarse_height * source_patch[0],
        config.coarse_width * source_patch[1],
    )
    output_shape = (
        config.coarse_height * output_patch[0],
        config.coarse_width * output_patch[1],
    )
    source_resolution = make_resolution(*source_shape)
    output_resolution = make_resolution(*output_shape)
    source = analytic_fields(
        coefficients,
        source_resolution,
        spectral_decay=config.spectral_decay,
    ).to(config.device)
    target = analytic_fields(
        coefficients,
        output_resolution,
        spectral_decay=config.spectral_decay,
    ).to(config.device)
    return source, target, source_resolution, output_resolution


def evaluate(
    model: CoarseLatentProbe,
    config: ProbeConfig,
    *,
    seed: int,
) -> dict[str, dict[str, float]]:
    model.eval()
    coefficients = make_coefficients(
        samples=config.eval_samples,
        channels=config.input_channels,
        seed=seed,
        device=config.device,
    )
    results: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for route_name, route in ROUTES.items():
            source, target, source_resolution, output_resolution = route_tensors(
                config, route, coefficients=coefficients
            )
            output, latent = model(source, source_resolution, output_resolution)
            output_patch = route[1]
            metrics = seam_metrics(output, target, *output_patch)
            metrics.update(
                spectral_ratios(
                    output,
                    target,
                    coarse_height=config.coarse_height,
                    coarse_width=config.coarse_width,
                )
            )
            metrics.update(
                {
                    "mse": float(F.mse_loss(output, target).cpu()),
                    "normalized_mse": float(
                        (
                            F.mse_loss(output, target)
                            / target.var(unbiased=False).clamp_min(1e-12)
                        ).cpu()
                    ),
                    "latent_rms": float(latent.square().mean().sqrt().cpu()),
                }
            )
            results[route_name] = metrics
    return results


def train(config: ProbeConfig) -> dict[str, object]:
    torch.manual_seed(config.seed)
    model = CoarseLatentProbe(config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    route_items = list(ROUTES.items())
    started = time.perf_counter()
    trajectory: list[dict[str, float | str]] = []
    report_every = max(1, config.steps // 10)
    model.train()
    for step in range(config.steps):
        route_name, route = route_items[step % len(route_items)]
        coefficients = make_coefficients(
            samples=config.batch_size,
            channels=config.input_channels,
            seed=config.seed * 1_000_003 + step + 1,
            device=config.device,
        )
        source, target, source_resolution, output_resolution = route_tensors(
            config, route, coefficients=coefficients
        )
        optimizer.zero_grad()
        output, _ = model(source, source_resolution, output_resolution)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % report_every == 0:
            trajectory.append(
                {
                    "step": float(step + 1),
                    "route": route_name,
                    "mse": float(loss.detach().cpu()),
                }
            )
    elapsed = time.perf_counter() - started
    evaluation = evaluate(model, config, seed=config.seed + 10_000_019)
    return {
        "config": asdict(config),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": elapsed,
        "evaluation": evaluation,
        "mean_normalized_mse": sum(
            route["normalized_mse"] for route in evaluation.values()
        )
        / len(evaluation),
        "trajectory": trajectory,
    }


def main() -> None:
    print(json.dumps(train(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
