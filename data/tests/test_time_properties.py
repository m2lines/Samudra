# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Averaging windows line up across sources, calendars and year boundaries.

OM4 averages over five-day windows that restart every January 1, so a leap
year's December 31 belongs to no window and the step across it lasts six days.
Forcing spliced in from another archive (wfo) and observations averaged onto
OM4's axis must use exactly OM4's windows, or the emulator learns from forcing
that is a step early or late.
"""

import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from hypothesis import given
from hypothesis import strategies as st
from ocean_preprocessing.obs_preprocessing.prepare import (
    _align_centered_5day,
    _coerce_time,
)
from ocean_preprocessing.simulation_preprocessing.gfdl_om4 import transplant_wfo

from samudra.invariants import require_no_violations, timeaxis
from tests.ocean_strategies import five_day_axes

pytestmark = pytest.mark.invariant(status="agreed")

WFO_ATTRS = {
    "cell_methods": "area:mean yh:mean xh:mean time: mean",
    "standard_name": "water_flux_into_sea_water",
    "units": "kg m-2 s-1",
}


def recipient_and_donor(stamps, bounds, shift: int = 0):
    n = len(stamps)
    grid = {"yh": [0.0, 1.0], "xh": [0.0, 1.0, 2.0]}
    recipient = xr.Dataset(
        {
            "thetao": (("time", "yh", "xh"), np.zeros((n, 2, 3))),
            "time_bnds": (("time", "nv"), bounds),
        },
        coords={"time": stamps, **grid},
    )
    ends = np.roll(bounds[:, 1], shift)
    donor = xr.Dataset(
        {"wfo": (("time", "yh", "xh"), np.arange(n * 6.0).reshape(n, 2, 3), WFO_ATTRS)},
        coords={"time": ends, **grid},
    )
    return recipient, donor


@given(five_day_axes(calendar="julian"))
def test_om4_axes_are_regular_across_leap_years(axis):
    """The generated axes match the published OM4 contract, so the tests below exercise it."""
    stamps, bounds = axis
    require_no_violations(
        timeaxis.om4_five_day_axis(stamps)
        + timeaxis.averaging_windows(stamps, bounds, leap_day_gaps=True)
    )


@given(five_day_axes())
def test_transplanted_wfo_keeps_each_window_value(axis):
    stamps, bounds = axis
    recipient, donor = recipient_and_donor(stamps, bounds)
    result = transplant_wfo(recipient, donor, donor_path="donor.zarr")
    np.testing.assert_array_equal(result.time.values, recipient.time.values)
    np.testing.assert_array_equal(result.wfo.values, donor.wfo.values)
    require_no_violations(
        timeaxis.windows_match(result.time_bnds.values, bounds, "thetao", "wfo")
    )


@given(five_day_axes(), st.sampled_from([-1, 1]))
def test_transplant_refuses_wfo_one_window_off(axis, shift):
    stamps, bounds = axis
    recipient, donor = recipient_and_donor(stamps, bounds, shift=shift)
    with pytest.raises(ValueError, match="upper bounds"):
        transplant_wfo(recipient, donor, donor_path="donor.zarr")


@given(
    five_day_axes(calendar="julian"), st.sampled_from(["noleap", "proleptic_gregorian"])
)
def test_transplant_refuses_wfo_in_another_calendar(axis, calendar):
    stamps, bounds = axis
    recipient, donor = recipient_and_donor(stamps, bounds)
    relabeled = [
        cftime.datetime(*t.timetuple()[:6], calendar=calendar)
        for t in donor.time.values
    ]
    donor = donor.assign_coords(time=relabeled)
    with pytest.raises(ValueError):
        transplant_wfo(recipient, donor, donor_path="donor.zarr")


@given(five_day_axes(calendar="julian", max_years=2), st.integers(0, 2**32 - 1))
def test_observation_means_cover_exactly_the_om4_window(axis, seed):
    """A centred five-day mean of daily data equals the mean over OM4's window, at every stamp."""
    stamps, bounds = axis
    targets = _coerce_time(stamps)
    first = targets[0].normalize() - pd.Timedelta(days=4)
    last = targets[-1].normalize() + pd.Timedelta(days=4)
    days = pd.date_range(first, last, freq="D")
    rng = np.random.default_rng(seed)
    daily = xr.Dataset(
        {"sst": ("time", rng.normal(size=days.size))}, coords={"time": days}
    )
    aligned = _align_centered_5day(daily, targets)

    day_bounds = np.stack([days.values, (days + pd.Timedelta(days=1)).values], axis=1)
    window_bounds = np.stack(
        [_coerce_time(bounds[:, 0]).values, _coerce_time(bounds[:, 1]).values], axis=1
    )
    require_no_violations(
        timeaxis.window_mean_preserved(
            daily.sst, day_bounds, aligned.sst, window_bounds, rtol=1e-9, atol=1e-12
        )
    )
