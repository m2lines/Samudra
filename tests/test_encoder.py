# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

import torch
from perceiver_pytorch import Perceiver

from ocean_emulators.constants import Lat, Lon
from ocean_emulators.models.modules.encoder import PerceiverEncoder, patch_from
from ocean_emulators.utils.ctx import GridContext

LATENT_DIM = 4
TOKEN_DIM = 8


def make_perceiver(*, num_latents=2):
    """Build a 1-D Perceiver matching the encoder's expected sequence shape."""
    return Perceiver(
        # num_freq_bands / max_freq are required positional kwargs but unused
        # when fourier_encode_data=False.
        num_freq_bands=0,
        max_freq=1.0,
        depth=2,
        input_axis=1,
        input_channels=TOKEN_DIM,
        fourier_encode_data=False,
        latent_dim=LATENT_DIM,
        num_latents=num_latents,
        num_classes=LATENT_DIM,
        weight_tie_layers=True,
        self_per_cross_attn=2,
    )


def make_encoder(
    prog_channels,
    boundary_channels,
    out_channels,
    patch_extent,
    max_patch_size=(8, 8),
):
    return PerceiverEncoder(
        prog_channels=prog_channels,
        boundary_channels=boundary_channels,
        out_channels=out_channels,
        token_dim=TOKEN_DIM,
        latent_dim=LATENT_DIM,
        patch_extent=patch_extent,
        max_patch_size=max_patch_size,
        perceiver=make_perceiver(),
    )


def make_resolution(x: torch.Tensor) -> tuple[Lat, Lon]:
    lat = torch.linspace(start=-90, end=90, steps=x.shape[-2])
    lon = torch.linspace(start=0, end=360, steps=x.shape[-1])
    return lat, lon


def make_ctx(prog: torch.Tensor) -> GridContext:
    """A minimal GridContext for encoder/FOMO tests; label_mask is a placeholder."""
    res = make_resolution(prog)
    mask = torch.ones(prog.shape[1], prog.shape[2], prog.shape[3], dtype=torch.bool)
    return GridContext(
        label_mask=mask,
        input_resolution_cpu=res,
        output_resolution_cpu=res,
    )


def test_makes_patches():
    prog = torch.randn(3, 7, 4, 8)
    boundary = torch.randn(3, 3, 4, 8)
    embed_dim = 4

    encoder = make_encoder(7, 3, embed_dim, (180, 180))
    patches = encoder(prog, boundary, make_ctx(prog))

    assert patches.shape == (3, embed_dim, 1, 2)


def test_makes_rectangular_patches():
    prog = torch.randn(1, 7, 4, 8)
    boundary = torch.randn(1, 3, 4, 8)
    embed_dim = 4

    encoder = make_encoder(7, 3, embed_dim, (180, 90))
    patches = encoder(prog, boundary, make_ctx(prog))

    assert patches.shape == (1, embed_dim, 1, 4)


def test_makes_patches__high_res():
    prog = torch.randn(1, 7, 14, 21)
    boundary = torch.randn(1, 3, 14, 21)
    embed_dim = 4

    encoder = make_encoder(7, 3, embed_dim, (90.0, 120.0))
    patches = encoder(prog, boundary, make_ctx(prog))

    assert patches.shape == (1, embed_dim, 2, 3)


def test_makes_patches__more_variables():
    prog = torch.randn(1, 17, 4, 8)
    boundary = torch.randn(1, 3, 4, 8)
    embed_dim = 4

    encoder = make_encoder(17, 3, embed_dim, (180, 180))
    patches = encoder(prog, boundary, make_ctx(prog))

    assert patches.shape == (1, embed_dim, 1, 2)


def test_cross_resolution_patches():
    """Prog and boundary at different resolutions still produce the same patch grid."""
    # Prog at 1/2 deg (8x16), boundary at 1 deg (4x8), patches of 90 degrees.
    prog = torch.randn(1, 7, 8, 16)
    boundary = torch.randn(1, 3, 4, 8)
    embed_dim = 4

    encoder = make_encoder(7, 3, embed_dim, (90.0, 90.0))
    patches = encoder(prog, boundary, make_ctx(prog))

    # 8 / patch_h(=4) = 2, 16 / patch_w(=4) = 4
    assert patches.shape == (1, embed_dim, 2, 4)


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
