import torch

from ocean_emulators.models.modules.encoder import PerceiverEncoder


def test_makes_patches():
    x = torch.randn(1, 10, 4, 8)

    patch_embed = PerceiverEncoder(
        in_channels=10,
        out_channels=4,
        patch_size=4,
        perceiver_depth=2,
        perceiver_impl="standard",
    )

    patches = patch_embed(x)

    assert patches.shape == (1, 4, 1, 2)


def test_makes_rectangular_patches():
    x = torch.randn(1, 10, 4, 8)

    patch_embed = PerceiverEncoder(
        in_channels=10,
        out_channels=4,
        patch_size=(4, 2),
        perceiver_depth=2,
        perceiver_impl="standard",
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
        in_channels=10,
        out_channels=5,
        patch_size=4,
        perceiver_depth=2,
        perceiver_impl="standard",
    )

    patches = patch_embed(x)

    assert patches.shape == (1, 5, 2, 4)


def test_makes_patches__more_variables():
    x = torch.randn(1, 20, 4, 8)

    patch_embed = PerceiverEncoder(
        in_channels=20,
        out_channels=5,
        patch_size=4,
        perceiver_depth=2,
        perceiver_impl="standard",
    )

    patches = patch_embed(x)

    assert patches.shape == (1, 5, 1, 2)
