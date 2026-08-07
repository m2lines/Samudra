# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Contracts that keep the ported SamudraMulti honest about what it computes.

The model can render a forecast two structurally different ways: decoding and
re-encoding a physical state at every step, or encoding once and advancing the
latent. Whichever one training uses, inference must use the same one.
"""

import pytest
import torch

from samudra.models.modules import (
    BoundaryEncoder,
    DirectPatchEncoder,
    ResampleProjectionDecoder,
)
from samudra.models.samudra_multi import SamudraMulti
from samudra.utils.ctx import GridContext

PROG_CHANNELS = 4
BOUNDARY_CHANNELS = 2
WIDTH = 8
H, W = 4, 8


def _resolution(height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.linspace(-90, 90, steps=height),
        torch.linspace(0, 360, steps=width),
    )


def _build(*, latent_autoregression: bool, project_before_resample: bool = False):
    """A minimal native-grid model: 1x1 encoder, identity-shaped processor, 1x1 decoder."""
    encoder = DirectPatchEncoder(
        in_channels=PROG_CHANNELS,
        out_channels=WIDTH,
        patch_extent=(1.0, 1.0),
        enforce_one_pixel_patch=False,
    )
    decoder = ResampleProjectionDecoder(
        in_channels=WIDTH,
        out_channels=PROG_CHANNELS,
        coordinate_resampling=project_before_resample,
        project_before_resample=project_before_resample,
    )
    return SamudraMulti(
        in_channels=PROG_CHANNELS,
        out_channels=PROG_CHANNELS,
        pred_residuals=False,
        last_kernel_size=3,
        pad="circular",
        add_3d_coordinates=None,
        encoder=encoder,
        processor=torch.nn.Conv2d(WIDTH, WIDTH, kernel_size=1),
        decoder=decoder,
        hist=0,
        checkpointing=None,
        gradient_detach_interval=1,
        use_bfloat16=False,
        boundary_encoder=BoundaryEncoder(
            boundary_channels=BOUNDARY_CHANNELS,
            processor_channels=WIDTH,
        ),
        latent_autoregression=latent_autoregression,
    )


@pytest.mark.parametrize("latent_autoregression", [False, True])
def test_rollout_state_matches_the_step_function_it_was_trained_with(
    latent_autoregression: bool,
):
    """`initialize_rollout` must hand back state in the same space the model steps in.

    With latent autoregression off, inference has to carry a physical state so it
    reproduces the decode/re-encode path that `forward_once` trains. With it on,
    inference carries the latent instead. Confusing the two silently evaluates a
    function the weights were never fit for.
    """
    torch.manual_seed(0)
    model = _build(latent_autoregression=latent_autoregression)
    prognostic = torch.randn(1, PROG_CHANNELS, H, W)
    ctx = GridContext(
        label_mask=torch.ones(PROG_CHANNELS, H, W, dtype=torch.bool),
        input_resolution_cpu=_resolution(H, W),
        output_resolution_cpu=_resolution(H, W),
    )

    state = model.initialize_rollout(prognostic, ctx)

    if latent_autoregression:
        assert state.shape[1] == WIDTH, "latent rollout must carry encoder width"
    else:
        assert state.shape == prognostic.shape, (
            "physical rollout must carry the prognostic state, not a latent"
        )


def test_latent_depth_training_refuses_residual_prediction():
    """The depth path decodes an absolute state; residual mode would add the input back.

    `initialize_rollout` already refuses this pairing, so training must too, or
    the two sides fit and evaluate different functions.
    """
    torch.manual_seed(0)
    model = _build(latent_autoregression=False)
    model.pred_residuals = True

    with pytest.raises(ValueError, match="pred_residuals must be false"):
        model.forward(train_data=None, loss_fn=None, processor_depth=1)


def test_cross_grid_decode_refuses_to_interpolate_without_the_channel_mask():
    """`project_before_resample` promises per-channel masked transport.

    No dataset populates `GridContext.input_mask` yet, so without this guard a
    cross-resolution decode quietly interpolates across land instead.
    """
    torch.manual_seed(0)
    model = _build(latent_autoregression=False, project_before_resample=True)
    latent = torch.randn(1, WIDTH, H, W)
    cross_grid_ctx = GridContext(
        label_mask=torch.ones(PROG_CHANNELS, H * 2, W * 2, dtype=torch.bool),
        input_resolution_cpu=_resolution(H, W),
        output_resolution_cpu=_resolution(H * 2, W * 2),
        input_mask=None,
    )

    with pytest.raises(ValueError, match="input_mask is unset"):
        model.decode(latent, _resolution(H, W), cross_grid_ctx)


def test_same_grid_decode_needs_no_mask_because_it_never_transports():
    """Same-grid decoding reduces to the learned 1x1 channel map, so it is exempt.

    This is what keeps the single-scale route usable while `input_mask` is unwired.
    """
    torch.manual_seed(0)
    model = _build(latent_autoregression=False, project_before_resample=True)
    latent = torch.randn(1, WIDTH, H, W)
    ctx = GridContext(
        label_mask=torch.ones(PROG_CHANNELS, H, W, dtype=torch.bool),
        input_resolution_cpu=_resolution(H, W),
        output_resolution_cpu=_resolution(H, W),
        input_mask=None,
    )

    decoded = model.decode(latent, _resolution(H, W), ctx)

    assert decoded.shape == (1, PROG_CHANNELS, H, W)
