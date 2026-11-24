# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
"""The tendency loss: GraphCast-style fixed per-channel residual weights.

GraphCast (Lam et al., 2023) trains on residuals scaled by the standard
deviation of each variable's one-step change: fixed weights, set by the data.
Unlike DynamicLoss, which reweights by the model's own running loss, these
weights do not move during training.

Synthetic setup: each channel's one-step change has an exact, chosen RMS, so
persistence error is known in closed form. State values have unit variance,
as after Samudra's per-channel normalization. Channel 0 stands in for a fast,
noisy field (velocity) and channel 1 for a slow one (temperature); their
change variances match those measured on quarter-degree OM4 validation data.
"""

import numpy as np
import pytest
import torch
import xarray as xr

from samudra.config import (
    TendencyLossConfig,
    build_loss_fn,
    load_normalized_persistence_rms,
)
from samudra.utils.ctx import BatchGrid
from samudra.utils.location import LocalLocation, UnresolvedLocation
from samudra.utils.loss import TendencyLoss, loss_fn_from_metric, tendency_loss_weights
from scripts.compute_tendency_stats import persistence_rms, select_time_range

B, H, W = 4, 12, 24
CPU = torch.device("cpu")
VARIANCE_RATIO = 470.0
# Channel 0: fast, noisy (velocity-like). Channel 1: slow (temperature-like).
# Normalized change variance ~0.10 for v and ~2.2e-4 for T on OM4 validation data.
CHANGE_RMS = torch.tensor([np.sqrt(0.1), np.sqrt(0.1 / VARIANCE_RATIO)])


def grid(channels: int) -> BatchGrid:
    lat = torch.linspace(-80, 80, H)
    lon = torch.linspace(0, 359, W)
    return BatchGrid(
        torch.ones(channels, H, W, dtype=torch.bool), (lat, lon), (lat, lon)
    )


def case(change_rms: torch.Tensor, seed: int = 0):
    """Return (state, target, change) with each channel's change RMS exact."""
    g = torch.Generator().manual_seed(seed)
    n = len(change_rms)
    state = torch.randn(B, n, H, W, generator=g)
    change = torch.randn(B, n, H, W, generator=g)
    change = change / change.square().mean(dim=(0, 2, 3), keepdim=True).sqrt()
    change = change * change_rms.view(1, -1, 1, 1)
    return state, state + change, change


def ratio(weights: torch.Tensor) -> float:
    """How much more the slow channel is weighted than the fast one."""
    return float(weights[1] / weights[0])


# -- TendencyLoss ---------------------------------------------------------------


def test_tendency_loss_scores_persistence_equally_in_every_channel():
    state, target, _ = case(CHANGE_RMS)
    loss = TendencyLoss(
        loss_fn_from_metric("mse"),
        weights=tendency_loss_weights(CHANGE_RMS, max_weight_ratio=None),
        device=CPU,
    )
    per_channel = loss(state, target, grid(2))
    torch.testing.assert_close(per_channel[0], per_channel[1])


def test_tendency_weights_are_inverse_change_variance_with_unit_mean():
    weights = tendency_loss_weights(CHANGE_RMS, max_weight_ratio=None)
    assert ratio(weights) == pytest.approx(VARIANCE_RATIO, rel=1e-5)
    assert float(weights.mean()) == pytest.approx(1.0)


def test_tendency_weights_cap_slowly_changing_channels():
    weights = tendency_loss_weights(CHANGE_RMS, max_weight_ratio=20.0)
    assert ratio(weights) == pytest.approx(20.0, rel=1e-5)


@pytest.mark.parametrize(
    "rms",
    [torch.tensor([1.0, 0.0]), torch.tensor([1.0, float("nan")]), torch.ones(2, 2)],
)
def test_tendency_weights_reject_invalid_statistics(rms):
    with pytest.raises(ValueError):
        tendency_loss_weights(rms, max_weight_ratio=None)


def test_tendency_loss_repeats_weights_across_output_steps():
    state, target, _ = case(CHANGE_RMS)
    two_steps_pred = torch.cat([state, state], dim=1)
    two_steps_target = torch.cat([target, target], dim=1)
    weights = tendency_loss_weights(CHANGE_RMS, max_weight_ratio=None)
    loss = TendencyLoss(loss_fn_from_metric("mse"), weights=weights, device=CPU)
    per_channel = loss(two_steps_pred, two_steps_target, grid(4))
    torch.testing.assert_close(per_channel[:2], per_channel[2:])


def graphcast_residual_mse(
    pred: torch.Tensor, target: torch.Tensor, last_input: torch.Tensor, diffs_std
) -> torch.Tensor:
    """Per-channel loss as in GraphCast's ``normalization.InputsAndResiduals``
    plus unweighted MSE: subtract the last input from target and prediction,
    divide by the per-channel change std (no location shift), then MSE.

    Latitude, pressure-level and per-variable weights are left out here; they
    are separate multiplicative factors in ``losses.weighted_loss_per_level``.
    """
    scale = diffs_std.view(1, -1, 1, 1)
    norm_pred = (pred - last_input) / scale
    norm_target = (target - last_input) / scale
    return (norm_pred - norm_target).square().mean(dim=(0, 2, 3))


def test_tendency_loss_equals_graphcast_residual_mse_up_to_a_constant():
    state, target, change = case(CHANGE_RMS)
    pred = state + change * torch.tensor([0.2, 0.7]).view(1, -1, 1, 1)
    reference = graphcast_residual_mse(pred, target, state, CHANGE_RMS)
    ours = TendencyLoss(
        loss_fn_from_metric("mse"),
        weights=tendency_loss_weights(CHANGE_RMS, max_weight_ratio=None),
        device=CPU,
    )(pred, target, grid(2))
    constant = ours / reference
    torch.testing.assert_close(constant[0], constant[1])


# -- Statistics and config wiring -------------------------------------------------


def test_persistence_rms_counts_land_as_zero_change():
    time = xr.date_range(
        "2000-01-01", periods=3, freq="5D", calendar="noleap", use_cftime=True
    )
    surface = np.zeros((3, 2, 2))
    surface[1:, :, 0] = 2.0  # wet column changes by 2 once, then stays
    deep = np.zeros((3, 2, 2))
    deep[1:, 0, 0] = 4.0
    data = xr.Dataset(
        {
            "thetao_0": (("time", "y", "x"), surface),
            "thetao_1": (("time", "y", "x"), deep),
            "zos": (("time", "y", "x"), surface),
            "mask_0": (("y", "x"), np.ones((2, 2), dtype=bool)),
            "mask_1": (("y", "x"), np.array([[True, False], [False, False]])),
        },
        coords={"time": time},
    )
    stats = persistence_rms(data, ["thetao_0", "thetao_1", "zos"], time_stride=1)
    # Two origins x four cells. Surface: change 2 in two cells at one origin.
    assert float(stats["thetao_0"]) == pytest.approx(np.sqrt(2 * 4 / 8))
    assert float(stats["zos"]) == float(stats["thetao_0"])
    # Level 1: change 4 in its only wet cell at one origin.
    assert float(stats["thetao_1"]) == pytest.approx(np.sqrt(16 / 8))


def test_persistence_rms_rejects_channels_that_never_change():
    time = xr.date_range(
        "2000-01-01", periods=2, freq="5D", calendar="noleap", use_cftime=True
    )
    data = xr.Dataset(
        {
            "zos": (("time", "y", "x"), np.zeros((2, 1, 1))),
            "mask_0": (("y", "x"), [[True]]),
        },
        coords={"time": time},
    )
    with pytest.raises(ValueError, match="never changes"):
        persistence_rms(data, ["zos"], time_stride=1)


def write_stats(path, values: dict[str, float], over: str = "full") -> LocalLocation:
    stats = xr.Dataset({k: xr.DataArray(v) for k, v in values.items()})
    stats.attrs["over"] = over
    stats.to_zarr(path)
    return LocalLocation(path=path.parent)


def test_build_tendency_loss_converts_physical_rms_to_normalized_units(tmp_path):
    root = write_stats(tmp_path / "rms.zarr", {"uo_0": 0.2, "thetao_0": 0.01})
    channels = ("uo_0", "thetao_0")
    state_stds = np.array([0.2, 0.5])  # normalized RMS: 1.0 and 0.02
    loss = build_loss_fn(
        TendencyLossConfig(
            persistence_rms_location=UnresolvedLocation(path="rms.zarr")
        ),
        device=CPU,
        num_channels=2,
        pad_mode="constant",
        data_root=root,
        channels=channels,
        state_stds=state_stds,
    )
    assert isinstance(loss, TendencyLoss)
    assert ratio(loss.loss_scale_per_channel()) == pytest.approx(1 / 0.02**2, rel=1e-5)


def test_missing_channel_statistics_fail_loudly(tmp_path):
    root = write_stats(tmp_path / "rms.zarr", {"uo_0": 0.2})
    with pytest.raises(KeyError, match="thetao_0"):
        load_normalized_persistence_rms(
            root.resolve(UnresolvedLocation(path="rms.zarr")),
            ("uo_0", "thetao_0"),
            np.ones(2),
            over="full",
        )


@pytest.mark.parametrize(
    "built, needed", [("wet", "full"), ("full", "wet"), (None, "wet")]
)
def test_statistics_built_with_other_averaging_fail_loudly(tmp_path, built, needed):
    # Full-grid and wet-only RMS differ by sqrt(wet fraction) per level; a run
    # given the wrong one would train normally with wrong channel weights.
    stats = xr.Dataset({"uo_0": xr.DataArray(0.2)})
    if built is not None:
        stats.attrs["over"] = built
    stats.to_zarr(tmp_path / "rms.zarr")
    with pytest.raises(ValueError, match=f"needs over='{needed}'"):
        load_normalized_persistence_rms(
            LocalLocation(path=tmp_path).resolve(UnresolvedLocation(path="rms.zarr")),
            ("uo_0",),
            np.ones(1),
            over=needed,
        )


@pytest.mark.parametrize("calendar", ["julian", "standard"])
def test_time_range_includes_whole_end_day_in_any_calendar(calendar):
    # Standard calendars decode to numpy datetimes, others to cftime.
    time = xr.date_range(
        "2000-01-01T12:00",
        periods=6,
        freq="5D",
        calendar=calendar,
        use_cftime=calendar != "standard",
    )
    data = xr.Dataset({"zos": ("time", np.arange(6.0))}, coords={"time": time})
    # Records on Jan 6, 11, 16 (stamped at noon); Jan 16 is the end day.
    selected = select_time_range(data, "2000-01-06", "2000-01-16")
    assert selected.zos.values.tolist() == [1.0, 2.0, 3.0]


def test_tendency_loss_requires_channel_context():
    with pytest.raises(ValueError, match="data_root"):
        build_loss_fn(
            TendencyLossConfig(
                persistence_rms_location=UnresolvedLocation(path="rms.zarr")
            ),
            device=CPU,
            num_channels=2,
            pad_mode="constant",
        )
