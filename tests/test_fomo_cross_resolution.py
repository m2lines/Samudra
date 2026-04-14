"""End-to-end test: FOMO.forward_once with mismatched prog/boundary resolutions."""

import pytest
import torch
from perceiver_pytorch import Perceiver
from perceiver_pytorch.perceiver_io import PerceiverIO

from ocean_emulators.models.fomo import FOMO
from ocean_emulators.models.modules import (
    PerceiverDecoder,
    PerceiverEncoder,
    UNetBackbone,
)
from ocean_emulators.models.modules.augment_input import Concat3dCoordinates
from ocean_emulators.models.modules.blocks import (
    BilinearUpsample,
    ConvNeXtBlock,
    CoreBlock,
)
from ocean_emulators.utils.ctx import GridContext

LATENT_DIM = 4
EMBED_DIM = 8
QUERIES_DIM = 16
PATCH_EXTENT = (90.0, 90.0)


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


def _make_perceiver(input_channels: int, depth: int = 2) -> Perceiver:
    return Perceiver(
        num_freq_bands=4,
        max_freq=10.0,
        depth=depth,
        input_axis=2,
        input_channels=input_channels,
        latent_dim=LATENT_DIM,
        num_latents=2,
        num_classes=LATENT_DIM,
        weight_tie_layers=True,
        self_per_cross_attn=2,
    )


def _make_fomo(
    prog_channels: int,
    boundary_channels: int,
    out_channels: int,
    add_3d_coordinates: bool = False,
) -> FOMO:
    encoder_prog_channels = prog_channels + (3 if add_3d_coordinates else 0)
    encoder = PerceiverEncoder(
        prog_channels=encoder_prog_channels,
        boundary_channels=boundary_channels,
        out_channels=EMBED_DIM,
        prog_latent_dim=LATENT_DIM,
        boundary_latent_dim=LATENT_DIM,
        patch_extent=PATCH_EXTENT,
        perceiver=_make_perceiver(encoder_prog_channels),
        boundary_perceiver=_make_perceiver(boundary_channels, depth=1),
    )
    processor = UNetBackbone(
        in_channels=EMBED_DIM,
        ch_width=[EMBED_DIM],
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
        dim=EMBED_DIM,
        queries_dim=QUERIES_DIM,
        logits_dim=out_channels,
        num_latents=4,
        latent_dim=LATENT_DIM,
        weight_tie_layers=True,
        decoder_ff=True,
    )
    decoder = PerceiverDecoder(
        in_channels=EMBED_DIM,
        out_channels=out_channels,
        patch_extent=PATCH_EXTENT,
        queries_dim=QUERIES_DIM,
        perceiver_io=perceiver_io,
        window_patches=None,
        context_patches=None,
    )
    return FOMO(
        in_channels=prog_channels + boundary_channels,
        out_channels=out_channels,
        pred_residuals=False,
        last_kernel_size=1,
        pad="circular",
        add_3d_coordinates=Concat3dCoordinates() if add_3d_coordinates else None,
        encoder=encoder,
        processor=processor,
        decoder=decoder,
        hist=0,
        checkpointing=None,
        gradient_detach_interval=0,
        use_bfloat16=False,
    )


def _make_resolution(h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.linspace(-90, 90, h), torch.linspace(0, 360, w)


def test_fomo_forward_once_cross_resolution():
    """FOMO produces output at prognostic resolution when boundary resolution differs."""
    prog_channels, boundary_channels, out_channels = 7, 3, 7
    # Prognostic: 1/4 degree (8x16), Boundary: 1 degree (2x4)
    prog_h, prog_w = 8, 16
    boundary_h, boundary_w = 2, 4

    model = _make_fomo(prog_channels, boundary_channels, out_channels)
    prog = torch.randn(2, prog_channels, prog_h, prog_w)
    boundary = torch.randn(2, boundary_channels, boundary_h, boundary_w)

    prog_res = _make_resolution(prog_h, prog_w)
    ctx = GridContext(
        label_mask=torch.ones(out_channels, prog_h, prog_w, dtype=torch.bool),
        input_resolution_cpu=prog_res,
        output_resolution_cpu=prog_res,
    )

    output = model.forward_once(prog, boundary, ctx)

    assert output.shape == (2, out_channels, prog_h, prog_w)
    assert torch.isfinite(output).all(), "Output contains NaN or Inf."


@pytest.mark.parametrize("add_3d_coordinates", [False, True])
def test_fomo_forward_once_mix_schedule(add_3d_coordinates: bool):
    """Mix schedule: prog at 1° input grid, output at ¼° grid."""
    prog_channels, boundary_channels, out_channels = 7, 3, 7
    # Input (prognostic) at 1 degree (4x8), output at 1/4 degree (16x32)
    input_h, input_w = 4, 8
    output_h, output_w = 16, 32

    model = _make_fomo(
        prog_channels,
        boundary_channels,
        out_channels,
        add_3d_coordinates=add_3d_coordinates,
    )
    prog = torch.randn(1, prog_channels, input_h, input_w)
    boundary = torch.randn(1, boundary_channels, input_h, input_w)

    input_res = _make_resolution(input_h, input_w)
    output_res = _make_resolution(output_h, output_w)
    ctx = GridContext(
        label_mask=torch.ones(out_channels, output_h, output_w, dtype=torch.bool),
        input_resolution_cpu=input_res,
        output_resolution_cpu=output_res,
    )

    output = model.forward_once(prog, boundary, ctx)

    assert output.shape == (1, out_channels, output_h, output_w)
    assert torch.isfinite(output).all(), "Output contains NaN or Inf."
