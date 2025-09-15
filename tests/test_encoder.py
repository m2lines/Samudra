import torch

from ocean_emulators.models.modules.encoder import PerceiverEncoder


def test_makes_patches():
    x = torch.randn(1, 10, 4, 8)

    patch_embed = PerceiverEncoder(
        in_channels=10,
        patch_size=4,
        embed_dim=4,
        perceiver_depth=2,
    )

    patches = patch_embed(x)

    assert patches.shape == (1, 1, 2, 4)


def test_makes_rectangular_patches():
    x = torch.randn(1, 10, 4, 8)

    patch_embed = PerceiverEncoder(
        in_channels=10,
        patch_size=(4, 2),
        embed_dim=4,
        perceiver_depth=2,
    )

    patches = patch_embed(x)

    assert patches.shape == (1, 1, 4, 4)


def test_makes_patches__high_res():
    x = torch.randn(1, 10, 8, 16)

    patch_embed = PerceiverEncoder(
        in_channels=10,
        patch_size=4,
        embed_dim=4,
        perceiver_depth=2,
    )

    patches = patch_embed(x)

    assert patches.shape == (1, 2, 4, 4)


def test_makes_patches__more_variables():
    x = torch.randn(1, 20, 4, 8)

    patch_embed = PerceiverEncoder(
        in_channels=20,
        patch_size=4,
        embed_dim=4,
        perceiver_depth=2,
    )

    patches = patch_embed(x)

    assert patches.shape == (1, 1, 2, 4)
