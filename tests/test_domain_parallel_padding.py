import torch

from ocean_emulators.config import BlockConfig, SamudraConfig, UNetBackboneConfig
from ocean_emulators.models.modules.blocks import ConvBlock, ConvNeXtBlock
from ocean_emulators.models.modules.padding import apply_spatial_pad
from ocean_emulators.shardtensor import validate_shardable


def test_constant_padding_refactor_preserves_convnext_output():
    torch.manual_seed(7)
    plain = ConvNeXtBlock(
        in_channels=8,
        out_channels=8,
        kernel_size=3,
        dilation=4,
        n_layers=1,
        pad="constant",
        upscale_factor=2,
        norm="group",
        group_norm_groups=4,
    )
    domain_parallel = ConvNeXtBlock(
        in_channels=8,
        out_channels=8,
        kernel_size=3,
        dilation=4,
        n_layers=1,
        pad="constant",
        domain_parallel=True,
        upscale_factor=2,
        norm="group",
        group_norm_groups=4,
    )
    domain_parallel.load_state_dict(plain.state_dict())

    x = torch.randn(1, 8, 32, 32)
    torch.testing.assert_close(plain(x), domain_parallel(x), atol=2e-6, rtol=0)


def test_constant_padding_refactor_preserves_conv_block_output():
    torch.manual_seed(8)
    plain = ConvBlock(
        in_channels=4,
        out_channels=4,
        kernel_size=3,
        dilation=2,
        n_layers=2,
        pad="constant",
    ).eval()
    domain_parallel = ConvBlock(
        in_channels=4,
        out_channels=4,
        kernel_size=3,
        dilation=2,
        n_layers=2,
        pad="constant",
        domain_parallel=True,
    ).eval()
    domain_parallel.load_state_dict(plain.state_dict())

    x = torch.randn(1, 4, 32, 32)
    torch.testing.assert_close(plain(x), domain_parallel(x), atol=2e-6, rtol=0)


def test_1088_is_shardable_on_a_2x2_cluster():
    validate_shardable(1088, 1088, (2, 2), num_downsamples=4)


def test_constant_padding_refactor_preserves_samudra_output():
    cfg = SamudraConfig(
        pred_residuals=False,
        last_kernel_size=3,
        pad="constant",
        checkpointing=None,
        use_bfloat16=False,
        unet=UNetBackboneConfig(
            ch_width=[4],
            dilation=[1],
            n_layers=[1],
            core_block=BlockConfig(
                block_type="conv_next_block",
                kernel_size=3,
                upscale_factor=2,
                norm="group",
                group_norm_groups=2,
            ),
        ),
    )
    wet = torch.ones(2, 32, 32)
    kwargs = dict(
        in_channels=4,
        out_channels=2,
        hist=0,
        wet=wet,
        area_weights=torch.ones_like(wet),
        static_data=None,
        lat=torch.linspace(-1.0, 1.0, 32),
        lon=torch.linspace(-1.0, 1.0, 32),
    )
    torch.manual_seed(9)
    plain = cfg.build(**kwargs)
    domain_parallel = cfg.build(**kwargs, domain_parallel=True)
    domain_parallel.load_state_dict(plain.state_dict())

    x = torch.randn(1, 4, 32, 32)
    torch.testing.assert_close(
        plain.predict_step(x), domain_parallel.predict_step(x), atol=2e-6, rtol=0
    )


def _forward_by_padding_the_input(block, x: torch.Tensor) -> torch.Tensor:
    """A ConvNeXt block's forward exactly as it ran before conv_padding.

    The two sibling tests above compare the plain block against the
    domain-parallel one, which was meaningful while those were two different
    code paths. With `pad="constant"` they are now the same path, so the
    claim that gave Conv2d the padding -- that zero-padding the input and
    convolving with padding=0 is the same arithmetic -- needs pinning against
    something that still does it the old way. This is that reference.
    """
    skip = None if block.disable_residual else block.skip_module(x)
    h = x
    for layer in block.convblock:
        if isinstance(layer, torch.nn.Conv2d) and layer.kernel_size[0] != 1:
            h = apply_spatial_pad(h, block.N_pad, "constant")
            h = torch.nn.functional.conv2d(
                h,
                layer.weight,
                layer.bias,
                stride=layer.stride,
                padding=0,
                dilation=layer.dilation,
            )
        else:
            h = layer(h)
    if block.disable_residual:
        return h
    assert skip is not None
    return skip + h


def test_giving_conv2d_the_padding_is_the_same_arithmetic():
    torch.manual_seed(11)
    block = ConvNeXtBlock(
        in_channels=8,
        out_channels=8,
        kernel_size=3,
        dilation=4,
        n_layers=1,
        pad="constant",
        upscale_factor=2,
        norm="group",
        group_norm_groups=4,
    ).eval()
    # The convs really are carrying it now, and the forward really has stopped
    # padding -- otherwise this would pass by padding twice.
    assert block.conv_padding == block.N_pad
    assert not block.pads_input

    x = torch.randn(1, 8, 32, 32)
    torch.testing.assert_close(
        block(x), _forward_by_padding_the_input(block, x), atol=2e-6, rtol=0
    )


def test_circular_padding_still_pads_the_input():
    """Wrap in x, zeros in y: Conv2d cannot express that, so it must not try."""
    block = ConvNeXtBlock(
        in_channels=4,
        out_channels=4,
        kernel_size=3,
        dilation=1,
        n_layers=1,
        pad="circular",
        upscale_factor=1,
        norm="group",
        group_norm_groups=2,
    )
    assert block.conv_padding == 0
    assert block.pads_input
