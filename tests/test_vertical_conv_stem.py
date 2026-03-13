import torch

from ocean_emulators.models.modules.vertical_conv_stem import VerticalConvStem


def test_vertical_conv_stem_with_depth_mixer_preserves_layout():
    stem = VerticalConvStem(
        num_3d_vars=2,
        num_depths=3,
        num_2d_vars=1,
        num_boundary_vars=2,
        hist=1,
        kernel_size=3,
        mid_channels=4,
        depth_mlp_hidden=5,
        shared_weights=False,
    )

    total_channels = (stem.hist + 1) * (
        stem.num_3d_vars * stem.num_depths + stem.num_2d_vars + stem.num_boundary_vars
    )
    x = torch.randn(2, total_channels, 4, 5)

    out = stem(x)

    assert out.shape == x.shape
    assert torch.equal(out[:, stem._idx_2d], x[:, stem._idx_2d])
    assert torch.equal(out[:, stem._boundary_slice(x.shape[1])], x[:, stem._boundary_slice(x.shape[1])])