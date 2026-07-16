import torch

from ocean_emulators.config import BlockConfig, SamudraConfig, UNetBackboneConfig
from ocean_emulators.models.modules.blocks import ConvBlock, ConvNeXtBlock
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
