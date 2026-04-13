"""End-to-end test: FOMO.forward_once with mismatched prog/boundary resolutions."""

import torch
from perceiver_pytorch import Perceiver
from perceiver_pytorch.perceiver_io import PerceiverIO

from ocean_emulators.models.fomo import FOMO
from ocean_emulators.models.modules import (
    PerceiverDecoder,
    PerceiverEncoder,
    UNetBackbone,
)
from ocean_emulators.models.modules.blocks import (
    BilinearUpsample,
    ConvNeXtBlock,
    CoreBlock,
)
from ocean_emulators.utils.ctx import GridContext


def _create_block(
    in_channels: int,
    out_channels: int,
    dilation: int,
    n_layers: int,
    pad: str,
    checkpoint_simple: bool,
) -> CoreBlock:
    return ConvNeXtBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        dilation=dilation,
        n_layers=n_layers,
        pad=pad,
        checkpoint_simple=checkpoint_simple,
    )


def _create_upsample(in_channels: int, out_channels: int) -> BilinearUpsample:
    return BilinearUpsample()


def test_fomo_forward_once_cross_resolution():
    """FOMO produces output at prognostic resolution when boundary resolution differs."""
    prog_channels = 7
    boundary_channels = 3
    out_channels = 7
    embed_dim = 8
    latent_dim = 4
    patch_extent = (90.0, 90.0)
    queries_dim = 16

    # Prognostic: 1/4 degree (8x16), Boundary: 1 degree (2x4)
    prog_h, prog_w = 8, 16
    boundary_h, boundary_w = 2, 4

    perceiver = Perceiver(
        num_freq_bands=4,
        max_freq=10.0,
        depth=2,
        input_axis=2,
        input_channels=prog_channels,
        latent_dim=latent_dim,
        num_latents=2,
        num_classes=latent_dim,
        weight_tie_layers=True,
        self_per_cross_attn=2,
    )
    boundary_perceiver = Perceiver(
        num_freq_bands=4,
        max_freq=10.0,
        depth=1,
        input_axis=2,
        input_channels=boundary_channels,
        latent_dim=latent_dim,
        num_latents=2,
        num_classes=latent_dim,
        weight_tie_layers=True,
        self_per_cross_attn=2,
    )
    encoder = PerceiverEncoder(
        prog_channels=prog_channels,
        boundary_channels=boundary_channels,
        out_channels=embed_dim,
        prog_latent_dim=latent_dim,
        boundary_latent_dim=latent_dim,
        patch_extent=patch_extent,
        perceiver=perceiver,
        boundary_perceiver=boundary_perceiver,
    )

    processor = UNetBackbone(
        in_channels=embed_dim,
        ch_width=[embed_dim],
        dilation=[1],
        n_layers=[1],
        pad="circular",
        create_block=_create_block,
        downsampling_block=torch.nn.Identity(),
        create_upsampling_block=_create_upsample,
        checkpointing=None,
    )

    perceiver_io = PerceiverIO(
        depth=2,
        dim=embed_dim,
        queries_dim=queries_dim,
        logits_dim=out_channels,
        num_latents=4,
        latent_dim=latent_dim,
        weight_tie_layers=True,
        decoder_ff=True,
    )
    decoder = PerceiverDecoder(
        in_channels=embed_dim,
        out_channels=out_channels,
        patch_extent=patch_extent,
        queries_dim=queries_dim,
        perceiver_io=perceiver_io,
        window_patches=None,
        context_patches=None,
    )

    model = FOMO(
        in_channels=prog_channels + boundary_channels,
        out_channels=out_channels,
        pred_residuals=False,
        last_kernel_size=1,
        pad="circular",
        add_3d_coordinates=None,
        encoder=encoder,
        processor=processor,
        decoder=decoder,
        hist=0,
        checkpointing=None,
        gradient_detach_interval=0,
        use_bfloat16=False,
    )

    batch = 2
    prog = torch.randn(batch, prog_channels, prog_h, prog_w)
    boundary = torch.randn(batch, boundary_channels, boundary_h, boundary_w)

    label_mask = torch.ones(out_channels, prog_h, prog_w, dtype=torch.bool)
    prog_res = (torch.linspace(-90, 90, prog_h), torch.linspace(0, 360, prog_w))
    ctx = GridContext(
        label_mask=label_mask,
        input_resolution_cpu=prog_res,
        output_resolution_cpu=prog_res,
    )

    output = model.forward_once(prog, boundary, ctx)

    assert output.shape == (batch, out_channels, prog_h, prog_w)
    assert torch.isfinite(output).all(), "Output contains NaN or Inf."
