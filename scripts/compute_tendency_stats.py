# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
"""Compute per-channel persistence RMS for the ``tendency`` loss.

For each prognostic channel this writes the root-mean-square one-step change,
in physical units. With ``--over full`` (the default) it averages over every
grid cell with land counted as zero change. That matches how the training loss
zeros land and averages over the full grid, so a persistence forecast scores
exactly 1 per channel under ``TendencyLoss``. With ``--over wet`` it averages
over wet cells only, which is the ``tau`` of the ``normalized_delta`` loss.
Use the training time range only.

Example:
    uv run python scripts/compute_tendency_stats.py \\
        --data s3://.../om4_quarterdeg_v2/OM4.zarr \\
        --output OM4_persistence_rms.zarr \\
        --start 1975-01-03 --end 2013-09-30 --time-stride 4

The output has one scalar variable per channel, like ``OM4_stds.zarr``.
Point ``loss.persistence_rms_location`` at it.
"""

import argparse
import datetime
import re

import cftime
import dask
import numpy as np
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


def persistence_rms(
    data: xr.Dataset, channels: list[str], time_stride: int, over: str = "full"
) -> xr.Dataset:
    """RMS of x[t+1] - x[t] over all grid cells, land counted as zero
    (``over="full"``), or over wet cells only (``over="wet"``).

    ``time_stride`` subsamples which origins are used; each difference is still
    one native time step.
    """
    if over not in ("full", "wet"):
        raise ValueError(f"over must be 'full' or 'wet', got {over!r}")
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
        count = wet_count if over == "wet" else change.size
        mean_square = sum_square / count
        if has_nan:
            raise ValueError(f"NaN one-step change in wet cells of {channel!r}")
        out[channel] = float(np.sqrt(mean_square))
        if out[channel] <= 0:
            raise ValueError(f"Channel {channel!r} never changes")
    return xr.Dataset({c: xr.DataArray(v) for c, v in out.items()})


def per_cell_rms(data: xr.Dataset, channels: list[str], time_stride: int) -> xr.Dataset:
    """RMS of x[t+1] - x[t] over time at every wet cell; NaN on land.

    One (y, x) map per channel, on the store's own grid and orientation.
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
        wet = wet_mask(data, channel)
        rms = np.sqrt(np.mean(np.square(change, dtype=np.float64), axis=0))
        rms, wet_values = dask.compute(rms, wet.data)
        if np.isnan(rms[wet_values]).any():
            raise ValueError(f"NaN one-step change in wet cells of {channel!r}")
        if not (rms[wet_values] > 0).any():
            raise ValueError(f"Channel {channel!r} never changes")
        out[channel] = xr.DataArray(np.where(wet_values, rms, np.nan), dims=wet.dims)
    return xr.Dataset(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="Flat OM4.zarr store")
    parser.add_argument("--output", required=True, help="Output zarr store")
    parser.add_argument("--start", required=True, help="First training date")
    parser.add_argument("--end", required=True, help="Last training date")
    parser.add_argument("--prognostic-vars-key", default="thermo_dynamic_all")
    parser.add_argument("--time-stride", type=int, default=1)
    parser.add_argument(
        "--over",
        choices=("full", "wet"),
        default="full",
        help="Average over the full grid (land as zero) or wet cells only",
    )
    parser.add_argument(
        "--per-cell",
        action="store_true",
        help="Write one RMS map per channel (over time, at each wet cell) instead",
    )
    args = parser.parse_args()

    channels = list(
        build_om4_layout(
            prognostic_vars_key=args.prognostic_vars_key
        ).prognostic_var_names
    )
    data = xr.open_zarr(args.data, chunks={})
    # Releases differ in calendar (e.g. julian, noleap); parse in the store's own.
    calendar = data.time.dt.calendar
    start = cftime.datetime.strptime(args.start, "%Y-%m-%d", calendar=calendar)
    end = cftime.datetime.strptime(args.end, "%Y-%m-%d", calendar=calendar)
    # Records are stamped mid-day; include the whole end day, as the trainer's
    # date-string slice does.
    end_exclusive = end + datetime.timedelta(days=1)
    data = data.sel(time=slice(start, None))
    data = data.isel(time=np.flatnonzero(data.time.values < end_exclusive))
    if data.sizes["time"] < 2:
        raise ValueError("Need at least two time steps in the training range")

    if args.per_cell:
        stats = per_cell_rms(data, channels, args.time_stride)
        stats.attrs["description"] = (
            "RMS one-step change over time at each wet cell, physical units, "
            "NaN on land"
        )
        stats.attrs["over"] = "per_cell"
    else:
        stats = persistence_rms(data, channels, args.time_stride, args.over)
        stats.attrs["description"] = (
            "RMS one-step change per channel, physical units, "
            + ("land counted as zero" if args.over == "full" else "wet cells only")
        )
        stats.attrs["over"] = args.over
    stats.attrs["time_range"] = f"{args.start} to {args.end}"
    stats.attrs["time_stride"] = args.time_stride
    stats.to_zarr(args.output, mode="w")
    for channel in channels:
        value = stats[channel]
        if args.per_cell:
            print(f"{channel}\twet RMS {float(np.sqrt(np.nanmean(value**2))):.6g}")
        else:
            print(f"{channel}\t{float(value):.6g}")


if __name__ == "__main__":
    main()
