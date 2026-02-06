import torch
from perceiver_pytorch import Perceiver

from ocean_emulators.constants import Lat, Lon
from ocean_emulators.models.modules.encoder import PerceiverEncoder, patch_from


def make_perceiver(in_channels, out_channels):
    return Perceiver(
        num_freq_bands=4,
        max_freq=1.0,
        depth=2,
        input_axis=2,
        input_channels=in_channels,
        latent_dim=3,
        num_latents=2,
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
