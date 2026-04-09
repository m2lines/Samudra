import torch
from perceiver_pytorch import Perceiver

from ocean_emulators.constants import Lat, Lon
from ocean_emulators.models.modules.encoder import PerceiverEncoder, patch_from

LATENT_DIM = 3


def make_perceiver(prog_channels, *, num_latents=2, max_freq=10.0):
    """Build a naive 2-D Perceiver for the prognostic stream.

    Uses ``input_axis=2`` so within-patch spatial structure is preserved
    via 2-D Fourier position encoding.
    """
    return Perceiver(
        num_freq_bands=4,
        max_freq=max_freq,
        depth=2,
        input_axis=2,
        input_channels=prog_channels,
        latent_dim=LATENT_DIM,
        num_latents=num_latents,
        num_classes=LATENT_DIM,
        weight_tie_layers=True,
        self_per_cross_attn=2,
    )


def make_resolution(x: torch.Tensor) -> tuple[Lat, Lon]:
    lat = torch.linspace(start=-90, end=90, steps=x.shape[-2])
    lon = torch.linspace(start=0, end=360, steps=x.shape[-1])
    return lat, lon


def test_makes_patches():
    prog = torch.randn(3, 7, 4, 8)
    boundary = torch.randn(3, 3, 4, 8)
    embed_dim = 4

    encoder = PerceiverEncoder(
        prog_channels=7,
        boundary_channels=3,
        out_channels=embed_dim,
        latent_dim=LATENT_DIM,
        patch_extent=(180, 180),
        perceiver=make_perceiver(7),
    )

    patches = encoder(prog, boundary, make_resolution(prog))

    assert patches.shape == (3, embed_dim, 1, 2)


def test_makes_rectangular_patches():
    prog = torch.randn(1, 7, 4, 8)
    boundary = torch.randn(1, 3, 4, 8)
    embed_dim = 4

    encoder = PerceiverEncoder(
        prog_channels=7,
        boundary_channels=3,
        out_channels=embed_dim,
        latent_dim=LATENT_DIM,
        patch_extent=(180, 90),
        perceiver=make_perceiver(7),
    )

    patches = encoder(prog, boundary, make_resolution(prog))

    assert patches.shape == (1, embed_dim, 1, 4)


def test_makes_patches__high_res():
    prog = torch.randn(1, 7, 14, 21)
    boundary = torch.randn(1, 3, 14, 21)
    embed_dim = 4

    encoder = PerceiverEncoder(
        prog_channels=7,
        boundary_channels=3,
        out_channels=embed_dim,
        latent_dim=LATENT_DIM,
        patch_extent=(90.0, 120.0),
        perceiver=make_perceiver(7),
    )

    patches = encoder(prog, boundary, make_resolution(prog))

    assert patches.shape == (1, embed_dim, 2, 3)


def test_makes_patches__more_variables():
    prog = torch.randn(1, 17, 4, 8)
    boundary = torch.randn(1, 3, 4, 8)
    embed_dim = 4

    encoder = PerceiverEncoder(
        prog_channels=17,
        boundary_channels=3,
        out_channels=embed_dim,
        latent_dim=LATENT_DIM,
        patch_extent=(180, 180),
        perceiver=make_perceiver(17),
    )

    patches = encoder(prog, boundary, make_resolution(prog))

    assert patches.shape == (1, embed_dim, 1, 2)


def test_cross_resolution_token_fuse():
    """Prog at 1/4 degree (8x16) and boundary at 1 degree (2x4) produce the same latent grid."""
    embed_dim = 4
    prog = torch.randn(1, 7, 8, 16)
    boundary = torch.randn(1, 3, 2, 4)

    encoder = PerceiverEncoder(
        prog_channels=7,
        boundary_channels=3,
        out_channels=embed_dim,
        latent_dim=LATENT_DIM,
        patch_extent=(90, 90),
        perceiver=make_perceiver(7),
    )

    prog_res = (
        torch.linspace(-90, 90, 8),
        torch.linspace(0, 360, 16),
    )
    patches = encoder(prog, boundary, prog_res)

    # Both grids with 90-degree patches → 2 lat patches, 4 lon patches
    assert patches.shape == (1, embed_dim, 2, 4)


def test_boundary_stream_influences_output():
    """Changing boundary data changes the encoder output, proving cross-attention is active."""
    embed_dim = 8
    prog = torch.randn(1, 5, 4, 8)
    boundary_a = torch.randn(1, 3, 4, 8)
    boundary_b = torch.randn(1, 3, 4, 8)

    encoder = PerceiverEncoder(
        prog_channels=5,
        boundary_channels=3,
        out_channels=embed_dim,
        latent_dim=LATENT_DIM,
        patch_extent=(180, 180),
        perceiver=make_perceiver(5),
    )
    encoder.eval()

    res = make_resolution(prog)
    out_a = encoder(prog, boundary_a, res)
    out_b = encoder(prog, boundary_b, res)

    assert not torch.allclose(out_a, out_b), (
        "Different boundary inputs should produce different outputs."
    )


def test_patch_from__full_globe():
    patch_h, patch_w = patch_from(
        patch_extent=(180.0, 360.0), input_height=4, input_width=8
    )
    assert patch_h == 4
    assert patch_w == 8


def test_patch_from__half_extent():
    patch_h, patch_w = patch_from(
        patch_extent=(90.0, 180.0), input_height=4, input_width=8
    )
    assert patch_h == 2
    assert patch_w == 4
