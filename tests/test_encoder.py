# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from einops import rearrange
from perceiver_pytorch import Perceiver
from torch import nn

from samudra.constants import Lat, Lon
from samudra.models.modules.encoder import (
    CanonicalResampleEncoder,
    DirectPatchEncoder,
    PerceiverEncoder,
    SpatialQueryPerceiver,
    patch_from,
)


def make_perceiver(in_channels, out_channels, *, num_latents=2, input_axis=2):
    return Perceiver(
        num_freq_bands=4,
        max_freq=1.0,
        depth=2,
        input_axis=input_axis,
        input_channels=in_channels,
        latent_dim=3,
        num_latents=num_latents,
        num_classes=out_channels,
        weight_tie_layers=True,
        self_per_cross_attn=2,
    )


def make_resolution(x: torch.Tensor) -> tuple[Lat, Lon]:
    lat = torch.linspace(start=-90, end=90, steps=x.shape[-2])
    lon = torch.linspace(start=0, end=360, steps=x.shape[-1])
    return lat, lon


class QueryProjection(nn.Module):
    def __init__(self, queries_dim: int, out_channels: int) -> None:
        super().__init__()
        self.projection = nn.Linear(queries_dim, out_channels)

    def forward(self, data: torch.Tensor, *, queries: torch.Tensor) -> torch.Tensor:
        batch_queries = queries.unsqueeze(0).expand(data.shape[0], -1, -1)
        # Retain a gradient path from every patch input for this lightweight stub.
        return self.projection(batch_queries) + data.mean() * 0


def test_canonical_resample_encoder_uses_fixed_finest_grid_and_has_gradients():
    source_resolution = (torch.tensor([-45.0, 45.0]), torch.arange(4) * 90.0)
    canonical_resolution = (torch.linspace(-67.5, 67.5, 4), torch.arange(8) * 45.0)
    encoder = CanonicalResampleEncoder(
        in_channels=10,
        out_channels=8,
        canonical_resolution=canonical_resolution,
        geometry_mode="none",
    )
    x = torch.randn(2, 10, 2, 4, requires_grad=True)

    encoded = encoder(x, source_resolution)

    assert encoded.shape == (2, 8, 4, 8)
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(
            encoder.output_resolution(source_resolution), canonical_resolution
        )
    )
    encoded.square().mean().backward()
    assert x.grad is not None
    assert encoder.projection.weight.grad is not None


def test_makes_patches():
    x = torch.randn(3, 10, 4, 8)

    patch_embed = PerceiverEncoder(
        in_channels=10,
        out_channels=4,
        patch_extent=(180, 180),
        perceiver=make_perceiver(10, 4),
    )

    patches = patch_embed(x, make_resolution(x))

    assert patches.shape == (3, 4, 1, 2)


def test_makes_rectangular_patches():
    x = torch.randn(1, 10, 4, 8)

    patch_embed = PerceiverEncoder(
        in_channels=10,
        out_channels=4,
        patch_extent=(180, 90),
        perceiver=make_perceiver(10, 4),
    )

    patches = patch_embed(x, make_resolution(x))

    assert patches.shape == (
        1,
        4,
        1,
        4,
    )


def test_perceiver_encoder_reports_patch_center_coordinates():
    x = torch.randn(1, 10, 4, 8)
    encoder = PerceiverEncoder(
        in_channels=10,
        out_channels=4,
        patch_extent=(90, 90),
        perceiver=make_perceiver(10, 4),
    )
    lat, lon = make_resolution(x)

    patch_lat, patch_lon = encoder.output_resolution((lat, lon))

    torch.testing.assert_close(patch_lat, lat.reshape(2, 2).mean(dim=1))
    torch.testing.assert_close(patch_lon, lon.reshape(4, 2).mean(dim=1))


def test_makes_patches__high_res():
    x = torch.randn(1, 10, 14, 21)

    patch_embed = PerceiverEncoder(
        in_channels=10,
        out_channels=4,
        patch_extent=(90.0, 120.0),
        perceiver=make_perceiver(10, 4),
    )

    patches = patch_embed(x, make_resolution(x))

    assert patches.shape == (1, 4, 2, 3)


def test_makes_patches__more_variables():
    x = torch.randn(1, 20, 4, 8)

    patch_embed = PerceiverEncoder(
        in_channels=20,
        out_channels=4,
        patch_extent=(180, 180),
        perceiver=make_perceiver(20, 4),
    )

    patches = patch_embed(x, make_resolution(x))

    assert patches.shape == (1, 4, 1, 2)


def test_spatial_queries_pack_ordered_outputs_as_processor_channels():
    x = torch.randn(2, 10, 6, 10, requires_grad=True)
    queries_dim = 7
    channels_per_query = 4
    spatial_perceiver = SpatialQueryPerceiver(
        query_shape=(3, 5),
        queries_dim=queries_dim,
        channels_per_query=channels_per_query,
        perceiver_io=QueryProjection(queries_dim, channels_per_query),
        num_freq_bands=4,
        max_freq=5,
    )
    patch_embed = PerceiverEncoder(
        in_channels=10,
        out_channels=spatial_perceiver.out_channels,
        patch_extent=(90, 180),
        perceiver=spatial_perceiver,
    )

    patches = patch_embed(x, make_resolution(x))

    assert spatial_perceiver.out_channels == 60
    assert patches.shape == (2, 60, 2, 2)
    patches.sum().backward()
    assert spatial_perceiver.query_offset.grad is not None


def test_direct_encoder_preserves_one_pixel_grid():
    x = torch.randn(2, 10, 6, 10, requires_grad=True)
    encoder = DirectPatchEncoder(
        in_channels=10,
        out_channels=12,
        patch_extent=(30, 36),
    )

    encoded = encoder(x, make_resolution(x))

    assert encoded.shape == (2, 12, 6, 10)
    encoded.sum().backward()
    assert encoder.projection.weight.grad is not None


def test_direct_encoder_can_leave_geometry_out_of_content():
    x = torch.randn(2, 10, 6, 10)
    encoder = DirectPatchEncoder(
        in_channels=10,
        out_channels=12,
        patch_extent=(30, 36),
        geometry_mode="none",
    )

    encoded = encoder(x, make_resolution(x))

    torch.testing.assert_close(encoded, encoder.projection(x))
    assert encoder.pos_embed is None
    assert encoder.scale_embed is None


def test_perceiver_encoder_can_leave_geometry_out_of_content():
    x = torch.randn(2, 10, 4, 8)
    perceiver = make_perceiver(10, 4)
    encoder = PerceiverEncoder(
        in_channels=10,
        out_channels=4,
        patch_extent=(180, 180),
        perceiver=perceiver,
        geometry_mode="none",
    )
    encoder.eval()
    patch_input = rearrange(
        x,
        "b v (h ph) (w pw) -> (b h w) ph pw v",
        ph=4,
        pw=4,
    )

    with torch.no_grad():
        expected = rearrange(
            perceiver(patch_input),
            "(b h w) c -> b c h w",
            b=2,
            h=1,
            w=2,
        )
        actual = encoder(x, make_resolution(x))

    torch.testing.assert_close(actual, expected)
    assert encoder.pos_embed is None
    assert encoder.scale_embed is None


def test_direct_encoder_rejects_spatial_compression():
    x = torch.randn(1, 10, 6, 10)
    encoder = DirectPatchEncoder(
        in_channels=10,
        out_channels=12,
        patch_extent=(60, 72),
    )

    with pytest.raises(ValueError, match="requires one-pixel patches"):
        encoder(x, make_resolution(x))


def test_patch_from__full_globe():
    # Full globe extent should equal grid dimensions
    patch_h, patch_w = patch_from(
        patch_extent=(180.0, 360.0), input_height=4, input_width=8
    )
    assert patch_h == 4
    assert patch_w == 8


def test_patch_from__half_extent():
    # Half the extent should give half the patch size
    patch_h, patch_w = patch_from(
        patch_extent=(90.0, 180.0), input_height=4, input_width=8
    )
    assert patch_h == 2
    assert patch_w == 4
