# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Time axes, calendars and averaging windows.

Averaging bugs cluster where windows meet calendar boundaries: the end of a year,
a leap day, or a change of calendar (OM4 is Julian, CM4 is no-leap, observations
are proleptic Gregorian). These checks work on cftime objects of any calendar.
"""

import datetime

import cftime
import numpy as np
import xarray as xr

from samudra.invariants.core import SECONDS_PER_DAY, Violation

OM4_STEPS_PER_YEAR = 73
OM4_STEP_DAYS = 5


def as_cftime(values, calendar: str | None = None) -> list[cftime.datetime]:
    """Time values as a list of cftime datetimes, from cftime or numpy datetime64."""
    array = np.asarray(values)
    if array.dtype.kind == "M":
        seconds = array.astype("datetime64[s]").astype(np.int64)
        return list(
            cftime.num2date(
                seconds, "seconds since 1970-01-01", calendar or "proleptic_gregorian"
            )
        )
    return list(array)


def calendar_of(values) -> str:
    """The calendar of a time axis; numpy datetime64 counts as proleptic Gregorian."""
    first = np.asarray(values).ravel()[0]
    if isinstance(first, np.datetime64):
        return "proleptic_gregorian"
    return first.calendar


def seconds_between(a, b) -> float:
    delta = b - a
    if isinstance(delta, np.timedelta64):
        return float(delta / np.timedelta64(1, "s"))
    if isinstance(delta, datetime.timedelta):
        return delta.total_seconds()
    raise TypeError(f"cannot take a duration from {type(delta).__name__}")


def strictly_increasing(time) -> list[Violation]:
    stamps = as_cftime(time)
    steps = [seconds_between(a, b) for a, b in zip(stamps, stamps[1:])]
    bad = [i for i, step in enumerate(steps) if step <= 0]
    if not bad:
        return []
    return [
        Violation(
            "time.strictly_increasing",
            f"{len(bad)} steps are zero or negative",
            {
                "time": bad[0] + 1,
                "previous": str(stamps[bad[0]]),
                "next": str(stamps[bad[0] + 1]),
            },
        )
    ]


def averaging_windows(
    time,
    time_bnds,
    step_days: float | None = OM4_STEP_DAYS,
    leap_day_gaps: bool = False,
) -> list[Violation]:
    """Each stamp is its window's midpoint and the windows tile time with no gaps or overlaps.

    Args:
        time: The time coordinate.
        time_bnds: Shape (time, 2): each window's start and end.
        step_days: Every window's expected length, or None to allow any length.
        leap_day_gaps: Allow exactly one uncovered day, December 31, at the end
            of each leap year. OM4's axis re-anchors every year to January 3
            with 73 five-day windows, which cover 365 days.
    """
    stamps = as_cftime(time)
    bounds = np.asarray(time_bnds)
    if bounds.shape != (len(stamps), 2):
        return [
            Violation(
                "time.bounds_shape",
                f"time_bnds has shape {bounds.shape}; expected ({len(stamps)}, 2)",
            )
        ]
    starts, ends = as_cftime(bounds[:, 0]), as_cftime(bounds[:, 1])
    violations = []
    for i, (stamp, start, end) in enumerate(zip(stamps, starts, ends)):
        length = seconds_between(start, end)
        if step_days is not None and length != step_days * SECONDS_PER_DAY:
            violations.append(
                Violation(
                    "time.window_length",
                    f"window [{start}, {end}] lasts {length / SECONDS_PER_DAY:g} days, "
                    f"not {step_days:g}",
                    {"time": i},
                )
            )
        if seconds_between(start, stamp) != seconds_between(stamp, end):
            violations.append(
                Violation(
                    "time.stamp_at_midpoint",
                    f"stamp {stamp} is not the midpoint of [{start}, {end}]",
                    {"time": i},
                )
            )
    calendar = calendar_of(time)
    for i in range(1, len(stamps)):
        gap = seconds_between(ends[i - 1], starts[i])
        if gap == 0:
            continue
        leap_day = (
            leap_day_gaps
            and gap == SECONDS_PER_DAY
            and cftime.is_leap_year(ends[i - 1].year, calendar)
            and (ends[i - 1].month, ends[i - 1].day, ends[i - 1].hour) == (12, 31, 0)
        )
        if not leap_day:
            kind = "gap" if gap > 0 else "overlap"
            violations.append(
                Violation(
                    f"time.windows_contiguous",
                    f"{kind} between [{starts[i - 1]}, {ends[i - 1]}] and "
                    f"[{starts[i]}, {ends[i]}]",
                    {"time": i},
                )
            )
    return violations


def om4_five_day_axis(time) -> list[Violation]:
    """The published OM4 axis: 73 five-day steps per year, each year starting on January 3, 12:00.

    365 = 73 x 5, so in a leap year the step from the year's last stamp to the
    next January 3 is 6 days. Any other spacing is a bug. Partial first and last
    years are allowed.
    """
    stamps = as_cftime(time)
    if not stamps:
        return [Violation("time.om4_axis", "empty time axis")]
    calendar = calendar_of(time)
    violations = strictly_increasing(time)
    by_year: dict[int, list] = {}
    for stamp in stamps:
        by_year.setdefault(stamp.year, []).append(stamp)
    years = sorted(by_year)
    for year in years:
        entries = by_year[year]
        first_expected = cftime.datetime(year, 1, 3, 12, calendar=calendar)
        if year != years[0] and entries[0] != first_expected:
            violations.append(
                Violation(
                    "time.om4_year_anchor",
                    f"{year} starts at {entries[0]}, not {first_expected}",
                    {"year": year},
                )
            )
        if year not in (years[0], years[-1]) and len(entries) != OM4_STEPS_PER_YEAR:
            violations.append(
                Violation(
                    "time.om4_steps_per_year",
                    f"{year} has {len(entries)} steps, not {OM4_STEPS_PER_YEAR}",
                    {"year": year},
                )
            )
    for i, (a, b) in enumerate(zip(stamps, stamps[1:])):
        days = seconds_between(a, b) / SECONDS_PER_DAY
        crosses_leap_year_end = b.year == a.year + 1 and cftime.is_leap_year(
            a.year, calendar
        )
        expected = OM4_STEP_DAYS + 1 if crosses_leap_year_end else OM4_STEP_DAYS
        if days != expected:
            violations.append(
                Violation(
                    "time.om4_step",
                    f"step from {a} to {b} is {days:g} days; expected {expected}",
                    {"time": i + 1},
                )
            )
    return violations


def windows_match(
    state_bnds, forcing_bnds, state_name: str = "state", forcing_name: str = "forcing"
) -> list[Violation]:
    """Forcing and state average over the same windows, in the same calendar.

    A forcing averaged over a window shifted by one step from the state's is the
    alignment bug that leaves the data plausible and the emulator wrong.
    """
    state = np.asarray(state_bnds)
    forcing = np.asarray(forcing_bnds)
    if state.shape != forcing.shape:
        return [
            Violation(
                "time.windows_match",
                f"{state_name} has windows of shape {state.shape}, {forcing_name} {forcing.shape}",
            )
        ]
    # xarray decodes standard calendars to datetime64 and others to cftime, so one
    # side may be each; compare them as cftime.
    state = np.array(as_cftime(state.ravel()), dtype=object).reshape(state.shape)
    forcing = np.array(as_cftime(forcing.ravel()), dtype=object).reshape(forcing.shape)
    if calendar_of(state) != calendar_of(forcing):
        return [
            Violation(
                "time.windows_match",
                f"{state_name} uses the {calendar_of(state)} calendar, "
                f"{forcing_name} the {calendar_of(forcing)} calendar",
            )
        ]
    mismatched = [
        i
        for i, (s, f) in enumerate(zip(state, forcing))
        if seconds_between(s[0], f[0]) != 0 or seconds_between(s[1], f[1]) != 0
    ]
    if not mismatched:
        return []
    i = mismatched[0]
    return [
        Violation(
            "time.windows_match",
            f"{len(mismatched)} windows differ; first: {state_name} "
            f"[{state[i][0]}, {state[i][1]}] vs {forcing_name} [{forcing[i][0]}, {forcing[i][1]}]",
            {"time": i},
        )
    ]


def window_mean_preserved(
    fine: xr.DataArray,
    fine_bnds,
    coarse: xr.DataArray,
    coarse_bnds,
    time_dim: str = "time",
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> list[Violation]:
    """Each coarse window's mean equals the duration-weighted mean of the fine windows inside it.

    Time averaging is linear, so averaging fine windows into coarse ones must
    reproduce the coarse values. Fine windows straddling a coarse boundary count
    in proportion to their overlap.
    """
    fine_b = [(as_cftime([s])[0], as_cftime([e])[0]) for s, e in np.asarray(fine_bnds)]
    violations = []
    coarse_b = [
        (as_cftime([s])[0], as_cftime([e])[0]) for s, e in np.asarray(coarse_bnds)
    ]
    for j, (cs, ce) in enumerate(coarse_b):
        weights = np.array(
            [max(0.0, seconds_between(max(fs, cs), min(fe, ce))) for fs, fe in fine_b]
        )
        total = weights.sum()
        length = seconds_between(cs, ce)
        if total != length:
            # Less than the window: a gap. More: fine windows overlap or repeat.
            violations.append(
                Violation(
                    "time.window_covered",
                    f"fine windows cover {total / SECONDS_PER_DAY:g} of the "
                    f"{length / SECONDS_PER_DAY:g} days in [{cs}, {ce}]",
                    {"time": j},
                )
            )
            continue
        inside = np.flatnonzero(weights)
        w = xr.DataArray(weights[inside] / total, dims=[time_dim])
        # skipna=False: a NaN inside the window makes the mean NaN, which then
        # matches only a NaN coarse value, instead of counting as zero.
        expected = (fine.isel({time_dim: inside}) * w).sum(time_dim, skipna=False)
        actual = coarse.isel({time_dim: j})
        bad = ~np.isclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
        if bad.any():
            difference = np.abs(np.asarray(actual - expected))
            worst = (
                f"worst by {float(np.nanmax(difference)):.3g}"
                if np.isfinite(difference).any()
                else "each is NaN on one side only"
            )
            violations.append(
                Violation(
                    "time.window_mean_preserved",
                    f"window [{cs}, {ce}]: {int(bad.sum())} values differ from the "
                    f"weighted mean of finer windows; {worst}",
                    {"time": j},
                )
            )
    return violations
