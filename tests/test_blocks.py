import torch

from ocean_emulators.models.modules.blocks import ConvNeXtBlock


def test_convnext_block_supports_group_norm():
    block = ConvNeXtBlock(
        in_channels=8,
        out_channels=8,
        kernel_size=3,
        dilation=1,
        n_layers=1,
        activation=None,
        pad="constant",
        upscale_factor=2,
        norm="group",
        group_norm_groups=16,
    )

    group_norm_layers = [m for m in block.convblock if isinstance(m, torch.nn.GroupNorm)]
    assert len(group_norm_layers) == 2
    assert all(layer.num_channels == 16 for layer in group_norm_layers)
    assert all(layer.num_groups == 16 for layer in group_norm_layers)

    x = torch.randn(2, 8, 12, 12)
    y = block(x)
    assert y.shape == (2, 8, 12, 12)


def test_convnext_block_group_norm_uses_compatible_group_count():
    block = ConvNeXtBlock(
        in_channels=7,
        out_channels=7,
        kernel_size=3,
        dilation=1,
        n_layers=1,
        activation=None,
        pad="constant",
        upscale_factor=2,
        norm="group",
        group_norm_groups=8,
    )

    group_norm_layers = [m for m in block.convblock if isinstance(m, torch.nn.GroupNorm)]
    assert len(group_norm_layers) == 2
    # 14 channels with requested 8 groups -> falls back to 7 groups.
    assert all(layer.num_channels == 14 for layer in group_norm_layers)
    assert all(layer.num_groups == 7 for layer in group_norm_layers)
