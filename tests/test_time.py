# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from samudra.utils.time import time_discontinuities


@pytest.mark.parametrize("days", [[], [0], [0, 1, 2, 3], [0, 5, 10, 15]])
def test_regular_times_have_no_discontinuities(days):
    times = np.datetime64("2000-01-01") + np.array(days).astype("timedelta64[D]")
    assert not time_discontinuities(times).any()


@pytest.mark.parametrize("days", [[0, 0, 5], [5, 0, 10]])
def test_unordered_times_fail(days):
    times = np.datetime64("2000-01-01") + np.array(days).astype("timedelta64[D]")
    with pytest.raises(ValueError, match="strictly increasing"):
        time_discontinuities(times)


def test_missing_timestamp_fails():
    with pytest.raises(ValueError, match="finite"):
        time_discontinuities(np.array(["2000-01-01", "NaT"], dtype="datetime64[D]"))


@pytest.mark.parametrize("calendar", ["julian", "numpy"])
@pytest.mark.parametrize("days", [[0, 6], [0, 10, 20], [0, 5, 10]])
def test_om4_intervals_are_checked_against_five_day_cadence(calendar, days):
    import datetime

    import cftime

    from samudra.constants import OM4_TIME_STEP

    if calendar == "julian":
        start = cftime.DatetimeJulian(1976, 12, 28, 12)
        times = np.array([start + datetime.timedelta(days=d) for d in days])
    else:
        times = np.datetime64("1976-12-28T12:00") + np.array(days).astype(
            "timedelta64[D]"
        )
    np.testing.assert_array_equal(
        time_discontinuities(times, OM4_TIME_STEP), np.diff(days) != 5
    )


@pytest.mark.parametrize("calendar", ["julian", "numpy"])
def test_irregular_times_require_declared_cadence(calendar):
    import datetime

    import cftime

    days = [0, 5, 11, 16]
    if calendar == "julian":
        start = cftime.DatetimeJulian(1976, 12, 23, 12)
        times = np.array([start + datetime.timedelta(days=d) for d in days])
    else:
        times = np.datetime64("1976-12-23T12:00") + np.array(days).astype(
            "timedelta64[D]"
        )
    with pytest.raises(ValueError, match="no expected time step is declared"):
        time_discontinuities(times)
