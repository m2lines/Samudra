# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

import torch
from perceiver_pytorch import Perceiver

from ocean_emulators.constants import Lat, Lon
from ocean_emulators.models.modules.encoder import PerceiverEncoder, patch_from

LATENT_DIM = 4


def make_perceiver(input_channels, *, num_latents=2, max_freq=10.0):
    """Build a naive 2-D Perceiver."""
    return Perceiver(
        num_freq_bands=4,
        max_freq=max_freq,
        depth=2,
        input_axis=2,
        input_channels=input_channels,
        latent_dim=LATENT_DIM,
        num_latents=num_latents,
        num_classes=LATENT_DIM,
        weight_tie_layers=True,
        self_per_cross_attn=2,
    )


def make_encoder(prog_channels, boundary_channels, out_channels, patch_extent):
    return PerceiverEncoder(
        prog_channels=prog_channels,
        boundary_channels=boundary_channels,
        out_channels=out_channels,
        prog_latent_dim=LATENT_DIM,
        boundary_latent_dim=LATENT_DIM,
        patch_extent=patch_extent,
        perceiver=make_perceiver(prog_channels),
        boundary_perceiver=make_perceiver(boundary_channels),
    )


def make_resolution(x: torch.Tensor) -> tuple[Lat, Lon]:
    lat = torch.linspace(start=-90, end=90, steps=x.shape[-2])
    lon = torch.linspace(start=0, end=360, steps=x.shape[-1])
    return lat, lon


def test_makes_patches():
    prog = torch.randn(3, 7, 4, 8)
    boundary = torch.randn(3, 3, 4, 8)
    embed_dim = 4

    encoder = make_encoder(7, 3, embed_dim, (180, 180))
    patches = encoder(prog, boundary, make_resolution(prog))

    assert patches.shape == (3, embed_dim, 1, 2)


def test_makes_rectangular_patches():
    prog = torch.randn(1, 7, 4, 8)
    boundary = torch.randn(1, 3, 4, 8)
    embed_dim = 4

    encoder = make_encoder(7, 3, embed_dim, (180, 90))
    patches = encoder(prog, boundary, make_resolution(prog))

    assert patches.shape == (1, embed_dim, 1, 4)


def test_makes_patches__high_res():
    prog = torch.randn(1, 7, 14, 21)
    boundary = torch.randn(1, 3, 14, 21)
    embed_dim = 4

    encoder = make_encoder(7, 3, embed_dim, (90.0, 120.0))
    patches = encoder(prog, boundary, make_resolution(prog))

    assert patches.shape == (1, embed_dim, 2, 3)


def test_makes_patches__more_variables():
    prog = torch.randn(1, 17, 4, 8)
    boundary = torch.randn(1, 3, 4, 8)
    embed_dim = 4

    encoder = make_encoder(17, 3, embed_dim, (180, 180))
    patches = encoder(prog, boundary, make_resolution(prog))

    assert patches.shape == (1, embed_dim, 1, 2)


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
