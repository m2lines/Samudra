# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Contracts for the native-grid SamudraMulti arrangement.

`project_before_resample` exists so that each prognostic channel is transported
with its own wet mask. Nothing populates `GridContext.input_mask` yet, so the
guard that refuses to interpolate without one is what keeps that promise from
degrading quietly into interpolation across land.
"""

import pytest
import torch

from samudra.models.modules import (
    BoundaryEncoder,
    NativeProjectionEncoder,
    ResampleProjectionDecoder,
)
from samudra.models.samudra_multi import SamudraMulti
from samudra.utils.ctx import GridContext

PROG_CHANNELS = 4
BOUNDARY_CHANNELS = 2
WIDTH = 8
H, W = 4, 8


def _resolution(height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Cell centers, so longitude does not duplicate the periodic seam."""
    return (
        torch.linspace(-90, 90, steps=height + 2)[1:-1],
        torch.arange(width, dtype=torch.float32) * (360.0 / width) + 180.0 / width,
    )


def _build(*, project_before_resample: bool):
    """A minimal native-grid model: 1x1 encoder, width-preserving processor, 1x1 decoder."""
    return SamudraMulti(
        in_channels=PROG_CHANNELS,
        out_channels=PROG_CHANNELS,
        pred_residuals=False,
        last_kernel_size=3,
        pad="circular",
        add_3d_coordinates=None,
        encoder=NativeProjectionEncoder(
            in_channels=PROG_CHANNELS,
            out_channels=WIDTH,
        ),
        processor=torch.nn.Conv2d(WIDTH, WIDTH, kernel_size=1),
        decoder=ResampleProjectionDecoder(
            in_channels=WIDTH,
            out_channels=PROG_CHANNELS,
            coordinate_resampling=project_before_resample,
            project_before_resample=project_before_resample,
        ),
        hist=0,
        checkpointing=None,
        gradient_detach_interval=1,
        use_bfloat16=False,
        boundary_encoder=BoundaryEncoder(
            boundary_channels=BOUNDARY_CHANNELS,
            processor_channels=WIDTH,
        ),
    )


def test_cross_grid_decode_refuses_to_interpolate_without_the_channel_mask():
    torch.manual_seed(0)
    model = _build(project_before_resample=True)
    cross_grid_ctx = GridContext(
        label_mask=torch.ones(PROG_CHANNELS, H * 2, W * 2, dtype=torch.bool),
        input_resolution_cpu=_resolution(H, W),
        output_resolution_cpu=_resolution(H * 2, W * 2),
        input_mask=None,
    )

    with pytest.raises(ValueError, match="input_mask is unset"):
        model.decode(torch.randn(1, WIDTH, H, W), _resolution(H, W), cross_grid_ctx)


def test_same_grid_decode_needs_no_mask_because_it_never_transports():
    """Same-grid decoding reduces to the learned 1x1 channel map, so it is exempt.

    This is what keeps the single-scale route usable while `input_mask` is unwired.
    """
    torch.manual_seed(0)
    model = _build(project_before_resample=True)
    ctx = GridContext(
        label_mask=torch.ones(PROG_CHANNELS, H, W, dtype=torch.bool),
        input_resolution_cpu=_resolution(H, W),
        output_resolution_cpu=_resolution(H, W),
        input_mask=None,
    )

    decoded = model.decode(torch.randn(1, WIDTH, H, W), _resolution(H, W), ctx)

    assert decoded.shape == (1, PROG_CHANNELS, H, W)


def test_masked_transport_excludes_land_from_interpolation():
    """A land cell must not contribute to any neighbouring ocean output value."""
    torch.manual_seed(0)
    model = _build(project_before_resample=True)
    mask = torch.ones(PROG_CHANNELS, H, W, dtype=torch.bool)
    mask[:, 0, 0] = False  # one land cell
    ctx = GridContext(
        label_mask=torch.ones(PROG_CHANNELS, H * 2, W * 2, dtype=torch.bool),
        input_resolution_cpu=_resolution(H, W),
        output_resolution_cpu=_resolution(H * 2, W * 2),
        input_mask=mask,
    )
    latent = torch.randn(1, WIDTH, H, W)

    masked = model.decode(latent, _resolution(H, W), ctx)
    # Make the land cell's value absurd; masked transport must ignore it.
    latent_poisoned = latent.clone()
    latent_poisoned[:, :, 0, 0] += 1e4
    poisoned = model.decode(latent_poisoned, _resolution(H, W), ctx)

    assert torch.allclose(masked, poisoned, atol=1e-4)


def test_state_encoder_ignores_boundary_forcing():
    """Forcing reaches the model only through its own encoder.

    The state representation the decoder inverts must not be a function of
    transient boundary values.
    """
    torch.manual_seed(0)
    model = _build(project_before_resample=False)
    ctx = GridContext(
        label_mask=torch.ones(PROG_CHANNELS, H, W, dtype=torch.bool),
        input_resolution_cpu=_resolution(H, W),
        output_resolution_cpu=_resolution(H, W),
    )
    prognostic = torch.randn(1, PROG_CHANNELS, H, W)

    first, _ = model.encode(prognostic, torch.randn(1, BOUNDARY_CHANNELS, H, W), ctx)
    second, _ = model.encode(prognostic, torch.zeros(1, BOUNDARY_CHANNELS, H, W), ctx)

    assert torch.equal(first, second)
