# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Metamorphic relations of the surface-initialized forecast model.

Each example builds a tiny model and randomizes every parameter. At construction
the ERA5 forcing encoder and geometry projections are zero, so forcing and geometry would
otherwise have no effect and several relations would hold vacuously.
"""

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from samudra.config import BlockConfig, UNetBackboneConfig
from samudra.models.surface_initialized import SurfaceInitializedModel
from samudra.observations.constants import (
    CONTEXT_CHANNELS,
    HISTORY_STEPS,
    MODEL_FORCING_CHANNELS,
    OBSERVATION_FORCING_CHANNELS,
    STATE_HISTORY_STEPS,
    SURFACE_NAMES,
)
from samudra.observations.replay import states_at

NAMES = ("thetao_0", "thetao_1", "so_0", "uo_0", "vo_0", "zos")
SURFACE = [NAMES.index(name) for name in SURFACE_NAMES]
LEVELS = 2  # U-Net levels; longitude shifts commute only in multiples of 2**LEVELS.
HEIGHT, WIDTH = 8, 16
SEEDS = st.integers(0, 2**31 - 1)
DOMAINS = st.sampled_from(["observation", "om4-family"])


@pytest.fixture(autouse=True)
def small_cpu_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def model(seed: int) -> SurfaceInitializedModel:
    torch.manual_seed(seed)
    network = SurfaceInitializedModel(
        NAMES,
        backbone=UNetBackboneConfig(
            ch_width=[4] * LEVELS,
            dilation=[1] * LEVELS,
            n_layers=[1] * LEVELS,
            core_block=BlockConfig(
                norm="instance", instance_norm_affine=True, upscale_factor=2
            ),
        ),
        activation_checkpointing=False,
    ).eval()
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.add_(0.1 * torch.randn_like(parameter))
    return network


def inputs(seed: int, domain: str, steps: int, batch: int = 1):
    """Global uniform grid with random land and random surface gaps."""
    generator = torch.Generator().manual_seed(seed)
    time = HISTORY_STEPS + steps
    lat = torch.linspace(-70, 70, HEIGHT).deg2rad()
    lon = (torch.arange(WIDTH) * 360 / WIDTH).deg2rad()
    lat, lon = torch.meshgrid(lat, lon, indexing="ij")
    xyz = torch.stack((lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()))
    season = torch.randn(batch, time, 2, 1, 1, generator=generator).expand(
        -1, -1, -1, HEIGHT, WIDTH
    )
    contexts = torch.cat((xyz.expand(batch, time, 3, HEIGHT, WIDTH), season), 2)
    assert contexts.shape[2] == CONTEXT_CHANNELS
    mask = torch.rand(len(NAMES), HEIGHT, WIDTH, generator=generator) < 0.8
    channels = (
        OBSERVATION_FORCING_CHANNELS
        if domain == "observation"
        else MODEL_FORCING_CHANNELS
    )
    shape = (batch, time, len(SURFACE_NAMES), HEIGHT, WIDTH)
    return dict(
        surface=torch.randn(shape, generator=generator),
        forcing=torch.randn(batch, time, channels, HEIGHT, WIDTH, generator=generator),
        contexts=contexts,
        mask=mask,
        validity=torch.rand(shape, generator=generator) < 0.7,
    )


def forecast(network, case, domain):
    with torch.no_grad():
        return network.forecast(**case, domain=domain)


@given(SEEDS, DOMAINS, st.integers(2, 4), st.data())
def test_continuation_from_any_split_reproduces_the_forecast(seed, domain, steps, data):
    """Oracle: the uninterrupted forecast, and the trajectory itself for the
    carried state pair. Replay relies on continuing from that pair being identical
    to never having stopped."""
    network, case = model(seed), inputs(seed, domain, steps)
    expected, initial = forecast(network, case, domain)
    split = data.draw(st.integers(1, steps - 1))
    future = {k: case[k][:, HISTORY_STEPS:] for k in ("forcing", "contexts")}
    with torch.no_grad():
        first, states = network.continue_forecast(
            initial,
            future["forcing"][:, :split],
            future["contexts"][:, :split],
            case["mask"],
            domain=domain,
        )
        second, _ = network.continue_forecast(
            states,
            future["forcing"][:, split:],
            future["contexts"][:, split:],
            case["mask"],
            domain=domain,
        )
    # The carried pair must be the trajectory's last two frames, not just consistent.
    assert torch.equal(states, states_at(initial, first, split))
    assert torch.equal(torch.cat((first, second), 1), expected)


@given(SEEDS, DOMAINS, st.integers(1, 3), st.data())
def test_predictions_never_depend_on_later_forcing_or_calendar(
    seed, domain, steps, data
):
    """Oracle: the unperturbed forecast. Changing forcing and context at forecast
    step j may change step j onward, but never an earlier prediction."""
    network, case = model(seed), inputs(seed, domain, steps)
    expected, _ = forecast(network, case, domain)
    j = data.draw(st.integers(0, steps - 1))
    changed = {k: v.clone() for k, v in case.items()}
    changed["forcing"][:, HISTORY_STEPS + j] += 3
    changed["contexts"][:, HISTORY_STEPS + j, 3:] += 3
    actual, _ = forecast(network, changed, domain)
    assert torch.equal(actual[:, :j], expected[:, :j])
    assert not torch.equal(actual[:, j], expected[:, j])


@given(SEEDS, DOMAINS)
def test_unobserved_surface_values_never_reach_the_model(seed, domain):
    """Oracle: the same forecast with arbitrary values, including NaN, written into
    every surface cell that is invalid or dry. Only supported cells are inputs."""
    network, case = model(seed), inputs(seed, domain, steps=1)
    expected, initial = forecast(network, case, domain)
    supported = case["validity"] & case["mask"][SURFACE]
    poisoned = dict(case, surface=case["surface"].masked_fill(~supported, float("nan")))
    actual, poisoned_initial = forecast(network, poisoned, domain)
    assert torch.equal(actual, expected)
    assert torch.equal(poisoned_initial, initial)


@given(SEEDS, DOMAINS)
def test_initial_state_keeps_measurements_and_dry_cells(seed, domain):
    """Oracle: the inputs themselves. In the two initial frames, measured wet
    surfaces are copied exactly, and every dry cell of every channel is zero."""
    network, case = model(seed), inputs(seed, domain, steps=1)
    _, initial = forecast(network, case, domain)
    frames = slice(HISTORY_STEPS - STATE_HISTORY_STEPS, HISTORY_STEPS)
    supported = (case["validity"] & case["mask"][SURFACE])[:, frames]
    assert torch.equal(
        initial[:, :, SURFACE][supported], case["surface"][:, frames][supported]
    )
    assert not initial.masked_select(~case["mask"].expand_as(initial)).any()


@pytest.mark.invariant(status="proposed")
@given(SEEDS, DOMAINS)
def test_batch_members_are_forecast_independently(seed, domain):
    """Oracle: each member's forecast alone. InstanceNorm normalizes per sample, so
    batching must not mix members (a BatchNorm block in train mode would)."""
    network = model(seed)
    pair = inputs(seed, domain, steps=2, batch=2)
    together, _ = forecast(network, pair, domain)
    for member in range(2):
        alone = {
            k: v if k == "mask" else v[member : member + 1] for k, v in pair.items()
        }
        separate, _ = forecast(network, alone, domain)
        torch.testing.assert_close(together[member : member + 1], separate)


@pytest.mark.invariant(status="proposed")
@given(SEEDS, DOMAINS, st.integers(1, WIDTH // 2**LEVELS - 1))
def test_circular_padding_makes_forecasts_commute_with_longitude_shifts(
    seed, domain, multiple
):
    """Oracle: the unshifted forecast, rolled. On a uniform global grid with
    circular padding, moving the longitude origin by a multiple of the pooling
    stride moves the forecast with it. Constant padding breaks this for any shift."""
    network, case = model(seed), inputs(seed, domain, steps=2)
    expected, _ = forecast(network, case, domain)
    shift = multiple * 2**LEVELS
    rolled = {k: torch.roll(v, shift, -1) for k, v in case.items()}
    actual, _ = forecast(network, rolled, domain)
    torch.testing.assert_close(
        actual, torch.roll(expected, shift, -1), atol=1e-4, rtol=1e-4
    )
