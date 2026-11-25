# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Artificial surface gaps and masked losses for the observation objective."""

import torch
from hypothesis import example, given
from hypothesis import strategies as st

from samudra.observations.constants import (
    HISTORY_STEPS,
    INITIAL_STATE_START,
    SURFACE_NAMES,
)
from samudra.observations.losses import (
    balanced_loss,
    channel_mse,
    completion_loss,
    corrupt_sample,
    structured_visibility,
)
from samudra.observations.sample_types import ObservationSample

SEEDS = st.integers(0, 2**31 - 1)


@st.composite
def fields(draw, channels=len(SURFACE_NAMES), time=None):
    """Seeded [batch, time, channel, y, x] values and a random validity mask."""
    batch = draw(st.integers(1, 2))
    time = draw(st.integers(1, 3)) if time is None else time
    height, width = draw(st.integers(2, 8)), draw(st.integers(2, 8))
    generator = torch.Generator().manual_seed(draw(SEEDS))
    shape = (batch, time, channels, height, width)
    values = torch.randn(shape, generator=generator)
    valid = torch.rand(shape, generator=generator) < draw(st.floats(0, 1))
    return values, valid


@given(fields(), SEEDS)
def test_visibility_only_hides_and_ignores_the_data(case, seed):
    """Oracle: a seed-only hiding pattern. Artificial gaps must be the original
    validity AND one fixed mask, the same at every time step, so they never reveal
    a missing value and never depend on which values happen to be observed."""
    _, valid = case
    everything = torch.ones_like(valid)
    pattern = structured_visibility(everything, seed)
    assert torch.equal(pattern, pattern[:, :1].expand_as(pattern))
    assert torch.equal(structured_visibility(valid, seed), valid & pattern)


@given(fields(time=HISTORY_STEPS), SEEDS)
def test_corruption_partitions_valid_cells_into_inputs_and_targets(case, seed):
    """Oracle: set algebra on the original validity. In the two initial-state frames
    every originally valid cell is either a visible input or a completion target,
    never both; hidden inputs are zero; the caller's sample is unchanged."""
    surface, valid = case
    surface = surface.masked_fill(~valid, float("nan"))
    sample = ObservationSample(surface=surface, validity=valid)
    before_surface, before_valid = surface.clone(), valid.clone()
    result = corrupt_sample(sample, seed)
    frames = slice(INITIAL_STATE_START, HISTORY_STEPS)
    visible, target = result["validity"][:, frames], result["completion_valid"]
    assert not (visible & target).any()
    assert torch.equal(visible | target, valid[:, frames])
    assert not result["surface"][~result["validity"]].any()
    assert torch.equal(
        result["surface"][result["validity"]], surface[result["validity"]]
    )
    assert torch.equal(sample["validity"], before_valid)
    torch.testing.assert_close(sample["surface"], before_surface, equal_nan=True)


@given(fields(), st.floats(0, 4), st.sampled_from(["completion", "channel"]))
# No supported cell at all: the empty-support branch once read NaN predictions.
@example(
    (torch.zeros(1, 1, 2, 2, 2), torch.zeros(1, 1, 2, 2, 2, dtype=torch.bool)),
    0.0,
    "completion",
)
def test_masked_losses_ignore_unsupported_cells_and_scale_quadratically(
    case, scale, loss
):
    """Oracle: the loss of a residual on its support. Replacing unsupported
    predictions and labels by NaN changes neither the loss nor its gradient's
    finiteness, and multiplying every supported residual by c multiplies the loss
    by c squared (so a perfect prediction scores zero)."""
    target, valid = case
    residual = torch.randn_like(target)
    latitude = torch.linspace(-80, 80, target.shape[-2])

    def evaluate(prediction, label):
        if loss == "completion":
            return completion_loss(prediction, label, valid, latitude)
        weights = valid * torch.cos(torch.deg2rad(latitude))[:, None]
        return channel_mse(prediction, label, weights).sum()

    prediction = (target + residual).requires_grad_()
    clean = evaluate(prediction, target)
    clean.backward()
    poisoned_prediction = torch.where(valid, target + residual, float("nan"))
    poisoned_prediction.requires_grad_()
    poisoned = evaluate(poisoned_prediction, target.masked_fill(~valid, float("nan")))
    poisoned.backward()
    torch.testing.assert_close(poisoned, clean)
    assert poisoned_prediction.grad is not None
    assert torch.isfinite(poisoned_prediction.grad).all()
    scaled = evaluate(target + scale * residual, target)
    torch.testing.assert_close(scaled, scale**2 * clean.detach(), rtol=1e-4, atol=1e-6)


NAMES = (
    *(f"thetao_{k}" for k in range(3)),
    *(f"so_{k}" for k in range(2)),
    "uo_0",
    "vo_0",
    "zos",
)


@given(st.permutations(range(len(NAMES))), SEEDS, st.booleans())
def test_balanced_loss_ignores_channel_order(order, seed, interior_only):
    """Oracle: the unpermuted loss. Variable groups are found by name, so
    reordering channels together with their names cannot change the loss."""
    generator = torch.Generator().manual_seed(seed)
    shape = (1, 2, len(NAMES), 3, 4)
    prediction = torch.randn(shape, generator=generator)
    target = torch.randn(shape, generator=generator)
    weights = (torch.rand(len(NAMES), 3, 4, generator=generator) < 0.7).float()
    order = list(order)
    torch.testing.assert_close(
        balanced_loss(
            prediction[:, :, order],
            target[:, :, order],
            weights[order],
            [NAMES[i] for i in order],
            interior_only,
        ),
        balanced_loss(prediction, target, weights, NAMES, interior_only),
    )
