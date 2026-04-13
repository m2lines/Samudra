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


def test_cross_resolution_token_fuse():
    """Prog at 1/4 degree (8x16) and boundary at 1 degree (2x4) produce the same latent grid."""
    embed_dim = 4
    prog = torch.randn(1, 7, 8, 16)
    boundary = torch.randn(1, 3, 2, 4)

    encoder = make_encoder(7, 3, embed_dim, (90, 90))

    prog_res = (
        torch.linspace(-90, 90, 8),
        torch.linspace(0, 360, 16),
    )
    patches = encoder(prog, boundary, prog_res)

    # Both grids with 90-degree patches → 2 lat patches, 4 lon patches
    assert patches.shape == (1, embed_dim, 2, 4)
    assert torch.isfinite(patches).all(), "Output contains NaN or Inf."


def test_latent_grid_mismatch_raises():
    """Misaligned latent grids between prog and boundary should raise."""
    import pytest

    embed_dim = 4
    # prog: 8x16 with patch_extent=(90,90) → ph=4,pw=4 → latent 2x4
    # boundary: 4x6 with same extent → ph=2,pw=2 → latent 2x3 (mismatch on lon)
    prog = torch.randn(1, 7, 8, 16)
    boundary = torch.randn(1, 3, 4, 6)

    encoder = make_encoder(7, 3, embed_dim, (90, 90))
    prog_res = (torch.linspace(-90, 90, 8), torch.linspace(0, 360, 16))

    with pytest.raises(AssertionError, match="Latent grid mismatch"):
        encoder(prog, boundary, prog_res)


def test_gradients_flow_to_both_streams():
    """Gradients flow from the output back to both prognostic and boundary inputs."""
    embed_dim = 4
    prog = torch.randn(1, 7, 4, 8, requires_grad=True)
    boundary = torch.randn(1, 3, 4, 8, requires_grad=True)

    encoder = make_encoder(7, 3, embed_dim, (180, 180))
    out = encoder(prog, boundary, make_resolution(prog))
    out.sum().backward()

    assert prog.grad is not None and prog.grad.abs().sum() > 0, (
        "Gradients must flow to prognostic input."
    )
    assert boundary.grad is not None and boundary.grad.abs().sum() > 0, (
        "Gradients must flow to boundary input."
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
