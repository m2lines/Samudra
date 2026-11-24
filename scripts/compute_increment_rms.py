# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
"""Compute per-channel increment RMS for the ``normalized_delta`` loss.

For each prognostic channel this writes the root-mean-square one-step change
over wet cells, in physical units: the ``tau`` of the ``normalized_delta``
loss. Use the training time range only.

Example:
    uv run python scripts/compute_increment_rms.py \\
        --data <data_root>/OM4.zarr \\
        --output <data_root>/OM4_increment_rms_wet.zarr \\
        --start 1975-01-03 --end 2013-09-30

The output has one scalar variable per channel, like ``OM4_stds.zarr``, and
records ``over="wet"``, which the loss checks. Point
``loss.increment_rms_location`` at it.
"""

import argparse
import datetime
import re

import cftime
import dask
import numpy as np
import pandas as pd
import xarray as xr

from samudra.constants import build_om4_layout

LEVEL_SUFFIX = re.compile(r"_(\d+)$")


def wet_mask(data: xr.Dataset, channel: str) -> xr.DataArray:
    """Each level's mask; single-level fields such as ``zos`` use the surface."""
    match = LEVEL_SUFFIX.search(channel)
    level = match.group(1) if match else "0"
    name = f"mask_{level}"
    if name not in data:
        raise KeyError(f"No wet mask {name!r} for channel {channel!r}")
    return data[name].astype(bool)


def increment_rms(
    data: xr.Dataset, channels: list[str], time_stride: int
) -> xr.Dataset:
    """RMS of x[t+1] - x[t] over each channel's wet cells.

    ``time_stride`` subsamples which origins are used; each difference is still
    one native time step.
    """
    if time_stride < 1:
        raise ValueError(f"time_stride must be positive, got {time_stride}")
    missing = [c for c in channels if c not in data]
    if missing:
        raise KeyError(f"Dataset is missing channels: {missing}")
    origins = np.arange(0, data.sizes["time"] - 1, time_stride)
    out = {}
    for channel in channels:
        field = data[channel]
        change = field.isel(time=origins + 1).data - field.isel(time=origins).data
        wet = wet_mask(data, channel).data
        change = np.where(wet, change, 0.0)
        # One pass over the data for all reductions.
        sum_square, wet_count, has_nan = dask.compute(
            np.sum(np.square(change, dtype=np.float64)),
            np.sum(wet) * len(origins),
            np.isnan(change).any(),
        )
        mean_square = sum_square / wet_count
        if has_nan:
            raise ValueError(f"NaN one-step change in wet cells of {channel!r}")
        out[channel] = float(np.sqrt(mean_square))
        if out[channel] <= 0:
            raise ValueError(f"Channel {channel!r} never changes")
    return xr.Dataset({c: xr.DataArray(v) for c, v in out.items()})


def select_time_range(data: xr.Dataset, start: str, end: str) -> xr.Dataset:
    """Records from the start day through the whole end day (YYYY-MM-DD).

    Records are stamped mid-day, so the end day is included in full, as the
    trainer's date-string slice does. Dates are parsed in the store's own
    calendar: cftime for non-standard calendars (OM4 releases are julian or
    noleap), numpy datetimes for standard ones.
    """
    times = data.time.values
    if isinstance(data.indexes["time"], xr.CFTimeIndex):
        calendar = data.time.dt.calendar
        first = cftime.datetime.strptime(start, "%Y-%m-%d", calendar=calendar)
        last = cftime.datetime.strptime(end, "%Y-%m-%d", calendar=calendar)
    else:
        first, last = pd.Timestamp(start), pd.Timestamp(end)
    end_exclusive = last + datetime.timedelta(days=1)
    return data.isel(time=np.flatnonzero((times >= first) & (times < end_exclusive)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="Flat OM4.zarr store")
    parser.add_argument("--output", required=True, help="Output zarr store")
    parser.add_argument("--start", required=True, help="First training date")
    parser.add_argument("--end", required=True, help="Last training date")
    parser.add_argument("--prognostic-vars-key", default="thermo_dynamic_all")
    parser.add_argument("--time-stride", type=int, default=1)
    args = parser.parse_args()

    channels = list(
        build_om4_layout(
            prognostic_vars_key=args.prognostic_vars_key
        ).prognostic_var_names
    )
    data = select_time_range(xr.open_zarr(args.data, chunks={}), args.start, args.end)
    if data.sizes["time"] < 2:
        raise ValueError("Need at least two time steps in the training range")

    stats = increment_rms(data, channels, args.time_stride)
    stats.attrs["description"] = (
        "RMS one-step change per channel over wet cells, physical units"
    )
    stats.attrs["over"] = "wet"
    stats.attrs["time_range"] = f"{args.start} to {args.end}"
    stats.attrs["time_stride"] = args.time_stride
    stats.to_zarr(args.output, mode="w")
    for channel in channels:
        print(f"{channel}\t{float(stats[channel]):.6g}")


if __name__ == "__main__":
    main()
