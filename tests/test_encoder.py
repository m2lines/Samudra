# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import torch

from samudra.constants import Lat, Lon
from samudra.models.modules import Perceiver, PerceiverIO
from samudra.models.modules.encoder import (
    DCTDetailEncoder,
    PerceiverEncoder,
    SpatialLatentGridEncoder,
    SpatialQueryPerceiver,
    dct_detail_basis,
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


def test_dct_detail_basis_is_orthonormal_and_phase_sensitive():
    basis = dct_detail_basis(2, 2, 3, device=torch.device("cpu"), dtype=torch.float32)
    torch.testing.assert_close(basis.T @ basis, torch.eye(4))

    constant = torch.ones(4)
    checkerboard = torch.tensor([1.0, -1.0, -1.0, 1.0])
    constant_coefficients = constant @ basis
    checkerboard_coefficients = checkerboard @ basis
    assert torch.allclose(constant_coefficients[1:], torch.zeros(3), atol=1e-6)
    assert checkerboard_coefficients[1:].abs().max() > 1


def test_dct_detail_encoder_preserves_patch_shape_and_gradients():
    encoder = DCTDetailEncoder(
        in_channels=2,
        out_channels=8,
        patch_extent=(90.0, 180.0),
        detail_count=3,
    )
    x = torch.randn(1, 2, 4, 4, requires_grad=True)
    encoded = encoder(x, make_resolution(x))
    assert encoded.shape == (1, 8, 2, 2)
    encoded.square().mean().backward()
    assert x.grad is not None


def test_spatial_query_encoder_emits_ordered_patch_features():
    spatial = SpatialQueryPerceiver(
        query_shape=(2, 2),
        queries_dim=4,
        channels_per_query=2,
        perceiver_io=PerceiverIO(
            depth=1,
            dim=20,
            queries_dim=4,
            logits_dim=2,
            num_latents=4,
            latent_dim=8,
            cross_heads=1,
            latent_heads=1,
            cross_dim_head=4,
            latent_dim_head=4,
            decoder_ff=True,
        ),
        num_freq_bands=4,
        max_freq=2,
    )
    encoder = PerceiverEncoder(
        in_channels=2,
        out_channels=8,
        patch_extent=(90.0, 180.0),
        perceiver=spatial,
    )
    x = torch.randn(1, 2, 4, 4, requires_grad=True)
    encoded = encoder(x, make_resolution(x))
    assert encoded.shape == (1, 8, 2, 2)
    encoded.mean().backward()
    assert x.grad is not None


def test_spatial_latent_grid_keeps_queries_as_spatial_tokens():
    spatial = SpatialQueryPerceiver(
        query_shape=(2, 2),
        queries_dim=4,
        channels_per_query=8,
        perceiver_io=PerceiverIO(
            depth=1,
            dim=20,
            queries_dim=4,
            logits_dim=8,
            num_latents=4,
            latent_dim=8,
            cross_heads=1,
            latent_heads=1,
            cross_dim_head=4,
            latent_dim_head=4,
            decoder_ff=True,
        ),
        num_freq_bands=4,
        max_freq=2,
    )
    encoder = SpatialLatentGridEncoder(
        in_channels=2,
        out_channels=8,
        patch_extent=(90.0, 180.0),
        spatial_perceiver=spatial,
    )
    x = torch.randn(1, 2, 8, 8, requires_grad=True)
    encoded = encoder(x, make_resolution(x))
    assert encoded.shape == (1, 8, 4, 4)
    assert encoder.output_patch_extent == (45.0, 90.0)
    encoded.mean().backward()
    assert x.grad is not None
