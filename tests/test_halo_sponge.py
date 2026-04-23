import torch

from ocean_emulators.models.modules.blocks import ConvNeXtBlock
from ocean_emulators.utils.loss import build_halo_sponge_spatial_weight, decomposed_mse


def test_convnext_block_supports_halo_sponge_padding():
    block = ConvNeXtBlock(
        in_channels=8,
        out_channels=8,
        kernel_size=3,
        dilation=1,
        n_layers=1,
        activation=None,
        pad="halo_sponge",
        upscale_factor=2,
        norm="batch",
    )

    x = torch.randn(2, 8, 12, 12)
    y = block(x)
    assert y.shape == (2, 8, 12, 12)


def test_halo_sponge_spatial_weight_ramps_inward():
    wet = torch.ones((1, 10, 10), dtype=torch.bool)
    spatial_weight = build_halo_sponge_spatial_weight(
        wet, num_halo=1, num_sponge=2
    )[0]

    assert spatial_weight[0, 0].item() == 0.0
    assert torch.isclose(spatial_weight[1, 1], torch.tensor(1.0 / 3.0))
    assert torch.isclose(spatial_weight[2, 2], torch.tensor(2.0 / 3.0))
    assert spatial_weight[3, 3].item() == 1.0


def test_halo_sponge_weighted_mse_preserves_unit_scale():
    wet = torch.ones((1, 10, 10), dtype=torch.bool)
    spatial_weight = build_halo_sponge_spatial_weight(
        wet, num_halo=1, num_sponge=2
    )

    pred = torch.ones((2, 1, 10, 10), dtype=torch.float32)
    target = torch.zeros_like(pred)
    loss = decomposed_mse(pred, target, wet, spatial_weight=spatial_weight)

    assert torch.allclose(loss, torch.ones_like(loss))
