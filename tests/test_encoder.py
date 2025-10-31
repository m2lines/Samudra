import torch
from perceiver_pytorch import Perceiver

from ocean_emulators.models.modules.encoder import PerceiverEncoder


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


def test_makes_patches():
    x = torch.randn(1, 10, 4, 8)

    patch_embed = PerceiverEncoder(
        in_channels=10,
        out_channels=4,
        patch_size=4,
        perceiver=make_perceiver(10, 4),
    )

    patches = patch_embed(x)

    assert patches.shape == (1, 4, 1, 2)


def test_makes_rectangular_patches():
    x = torch.randn(1, 10, 4, 8)

    patch_embed = PerceiverEncoder(
        in_channels=10,
        out_channels=4,
        patch_size=(4, 2),
        perceiver=make_perceiver(10, 4),
    )

    patches = patch_embed(x)

    assert patches.shape == (
        1,
        4,
        1,
        4,
    )


def test_makes_patches__high_res():
    x = torch.randn(1, 10, 8, 16)

    patch_embed = PerceiverEncoder(
        in_channels=10, out_channels=5, patch_size=4, perceiver=make_perceiver(10, 5)
    )

    patches = patch_embed(x)

    assert patches.shape == (1, 5, 2, 4)


def test_makes_patches__more_variables():
    x = torch.randn(1, 20, 4, 8)

    patch_embed = PerceiverEncoder(
        in_channels=20,
        out_channels=5,
        patch_size=4,
        perceiver=make_perceiver(20, 5),
    )

    patches = patch_embed(x)

    assert patches.shape == (1, 5, 1, 2)
