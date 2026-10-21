# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Explicit five-day support and monthly batches for observational transfer."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = {
    "train": ("1993-05", "2013-07"),
    "validation": ("2013-11", "2014-07"),
    "test": ("2015-01", "2022-12"),
}


def month_intervals(month):
    """Nineteen historical bins and 6/7 forecast bins, as half-open day intervals."""
    origin = pd.Timestamp(month)
    if origin.day != 1 or origin != origin.normalize():
        raise ValueError("A forecast origin must be midnight on the first of a month")
    next_month = origin + pd.offsets.MonthBegin()
    leads = int(np.ceil((next_month - origin).days / 5))
    starts = pd.date_range(
        origin - pd.Timedelta(days=95), periods=19 + leads, freq="5D"
    )
    ends = starts + pd.Timedelta(days=5)
    overlap = np.array(
        [
            max(0, (min(end, next_month) - max(start, origin)).days)
            for start, end in zip(starts[19:], ends[19:], strict=True)
        ],
        dtype=np.float32,
    )
    if overlap.sum() != (next_month - origin).days:
        raise ValueError("Monthly integration does not cover the calendar month")
    return starts, ends, overlap / overlap.sum()


def strict_mean(values):
    """No invented labels from incomplete temporal support."""
    valid = np.isfinite(values).all(axis=0)
    mean = np.where(np.isfinite(values), values, 0).mean(axis=0)
    return np.where(valid, mean, np.nan).astype(np.float32)


class ObservationArchive:
    def __init__(self, root):
        self.root = Path(root)
        self.grid = dict(np.load(self.root / "grid.npz"))

    @lru_cache(maxsize=128)
    def day(self, product, date):
        stamp = pd.Timestamp(date)
        path = (
            self.root
            / product
            / str(stamp.year)
            / (stamp.strftime("%Y-%m-%d") + ".npz")
        )
        with np.load(path) as saved:
            return saved["values"]

    def interval(self, product, start, end):
        dates = pd.date_range(start, end - pd.Timedelta(days=1), freq="D")
        if len(dates) != 5:
            raise ValueError(
                "Model input/output intervals must contain exactly five days"
            )
        return strict_mean(
            np.stack([self.day(product, str(day.date())) for day in dates])
        )

    def sample(self, month):
        starts, ends, month_weights = month_intervals(month)
        surface, atmosphere, velocity = [], [], []
        for start, end in zip(starts, ends, strict=True):
            adt = self.interval("adt", start, end)
            surface.append(np.concatenate([self.interval("sst", start, end), adt[:1]]))
            velocity.append(adt[1:])
            atmosphere.append(self.interval("era5", start, end))
        stamp = pd.Timestamp(month)
        with np.load(
            self.root / "iap" / str(stamp.year) / (stamp.strftime("%Y-%m") + ".npz")
        ) as saved:
            interior = saved["values"]
            ohc = saved["ohc"]
        return dict(
            surface=np.stack(surface),
            atmosphere=np.stack(atmosphere),
            velocity=np.stack(velocity),
            interior=interior,
            ohc=ohc,
            month_weights=month_weights,
            midpoints=np.array(
                [(s + pd.Timedelta(days=2.5)).isoformat() for s in starts]
            ),
            origin=np.array(stamp.isoformat()),
        )

    def materialize(self, split, output):
        """Fail loudly on incomplete source coverage; no silent sample rejection."""
        from samudra.observations.prepare import save_atomic

        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        for month in pd.period_range(*SPLITS[split], freq="M"):
            path = output / (str(month) + ".npz")
            sample = self.sample(str(month))
            save_atomic(path, **sample)


def context_planes(lat, lon, midpoints):
    """Spherical geography and seasonal phase at each interval midpoint."""
    latitude, longitude = np.meshgrid(np.deg2rad(lat), np.deg2rad(lon), indexing="ij")
    xyz = np.stack(
        [
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        ]
    )
    result = []
    for midpoint in pd.DatetimeIndex(midpoints):
        year_start = pd.Timestamp(year=midpoint.year, month=1, day=1)
        next_year = pd.Timestamp(year=midpoint.year + 1, month=1, day=1)
        phase = 2 * np.pi * (midpoint - year_start) / (next_year - year_start)
        season = np.broadcast_to(
            np.array([np.sin(phase), np.cos(phase)])[:, None, None],
            (2,) + latitude.shape,
        )
        result.append(np.concatenate([xyz, season]))
    return np.array(result, dtype=np.float32)


def fit_statistics(samples, grid, output):
    """Training-example-distribution statistics; repeated history weighting is explicit."""
    from samudra.observations.prepare import save_atomic

    lat, mask = grid["lat"], grid["mask"]
    domain = mask[0].astype(bool) & (np.abs(lat[:, None]) <= 60)
    area = np.cos(np.deg2rad(lat))[:, None] * domain
    h, w = mask.shape[-2:]
    climatology_sum = np.zeros((12, 2, h, w), dtype=np.float64)
    climatology_count = np.zeros_like(climatology_sum)
    moments = {
        "surface": np.zeros((3, 2)),
        "atmosphere": np.zeros((3, 8)),
        "interior": np.zeros((3, 28)),
    }
    count = 0
    for path in sorted(Path(samples).glob("*.npz")):
        month = path.stem
        if not SPLITS["train"][0] <= month <= SPLITS["train"][1]:
            raise ValueError(f"Non-training month in normalization inputs: {month}")
        with np.load(path) as sample:
            for name, storage in moments.items():
                values = sample[name].astype(np.float64)
                if name == "interior":
                    values = values.reshape(1, 28, h, w)
                valid = np.isfinite(values)
                weights = valid * area
                safe = np.where(valid, values, 0)
                storage[0] += weights.sum((0, 2, 3))
                storage[1] += (safe * weights).sum((0, 2, 3))
                storage[2] += (safe**2 * weights).sum((0, 2, 3))
            for field, date in zip(
                sample["surface"], pd.DatetimeIndex(sample["midpoints"]), strict=True
            ):
                valid = np.isfinite(field)
                climatology_sum[date.month - 1] += np.where(valid, field, 0)
                climatology_count[date.month - 1] += valid
        count += 1
    if count != 243:
        raise ValueError(f"Expected 243 accepted training months, found {count}")
    result = {}
    for name, (weight, total, square) in moments.items():
        if not (weight > 0).all():
            raise ValueError(f"Empty normalization channel in {name}")
        mean = total / weight
        variance = square / weight - mean**2
        if not (variance > 0).all():
            raise ValueError(f"Constant or invalid normalization channel in {name}")
        result[name + "_mean"] = mean.astype(np.float32)
        result[name + "_std"] = np.sqrt(variance).astype(np.float32)
    climatology = np.divide(
        climatology_sum,
        climatology_count,
        out=np.full_like(climatology_sum, np.nan),
        where=climatology_count > 0,
    )
    # Entirely unobserved input cells use a training-only monthly ocean mean.
    # They always retain validity=0 and never become labels.
    for month_index in range(12):
        for channel in range(2):
            field = climatology[month_index, channel]
            valid = np.isfinite(field) & domain
            fallback = np.sum(np.where(valid, field, 0) * area) / np.sum(valid * area)
            field[~np.isfinite(field)] = fallback
    result["surface_climatology"] = climatology.astype(np.float32)
    result["training_month_count"] = np.array(count)
    save_atomic(Path(output), **result)


def trajectory(archive, origin, *, target_month=None, steps=None, split="train"):
    """Read a small trajectory with five-day bins anchored at its original origin.

    Monthly replay recomputes the boundary-straddling bin. Its two-state snapshot
    must precede ``start_lead``; merely appending adjacent monthly files is wrong.
    """
    from samudra.observations.intervals import calendar_weights, replay_month

    origin = pd.Timestamp(origin)
    if (target_month is None) == (steps is None):
        raise ValueError("Specify exactly one target_month or steps")
    if target_month is not None:
        first = pd.Timestamp(target_month)
        start, stop, next_start = replay_month(origin, first)
        last = first + pd.offsets.MonthBegin()
    else:
        if steps < 1 or steps > 73:
            raise ValueError("Annual trajectories require 1..73 steps")
        first = origin
        start, stop, next_start = 0, int(steps), int(steps)
        last = origin + pd.Timedelta(days=5 * stop)
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split}")
    lower = pd.Timestamp(SPLITS[split][0])
    # July's final five-day interval ends August 5 in the established samples.
    upper = (
        pd.Timestamp("2013-08-05")
        if split == "train"
        else pd.Timestamp("2015-01-01")
        if split == "validation"
        else pd.Timestamp("2023-01-01")
    )
    if stop > 73:
        raise ValueError("Replay exceeds one year")
    forecast_end = origin + pd.Timedelta(days=5 * stop)
    if (
        origin < lower
        or forecast_end > upper
        or (first.year, first.month)
        > (
            (2013, 7)
            if split == "train"
            else (2014, 12)
            if split == "validation"
            else (2022, 12)
        )
    ):
        raise ValueError("Trajectory crosses the declared split boundary")
    starts = pd.date_range(
        origin + pd.Timedelta(days=5 * (start - 19)),
        periods=19 + stop - start,
        freq="5D",
    )
    ends = starts + pd.Timedelta(days=5)
    surface, atmosphere, velocity = [], [], []
    for a, b in zip(starts, ends, strict=True):
        adt = archive.interval("adt", a, b)
        surface.append(np.concatenate([archive.interval("sst", a, b), adt[:1]]))
        atmosphere.append(archive.interval("era5", a, b))
        velocity.append(adt[1:])
    months, weights, interiors, ohcs = [], [], [], []
    for month in pd.period_range(first, last - pd.Timedelta(days=1), freq="M"):
        a, b = month.start_time, (month + 1).start_time
        if a < starts[19] or b > ends[-1]:
            continue
        path = archive.root / "iap" / str(a.year) / f"{month}.npz"
        with np.load(path) as saved:
            interiors.append(saved["values"])
            ohcs.append(saved["ohc"])
        months.append(str(month))
        weights.append(calendar_weights(starts[19:], ends[19:], a, b))
    if not months:
        raise ValueError("Trajectory contains no complete monthly target")
    return dict(
        surface=np.stack(surface),
        atmosphere=np.stack(atmosphere),
        velocity=np.stack(velocity),
        interior=interiors[0],
        ohc=ohcs[0],
        month_weights=weights[0],
        monthly_targets=np.stack(interiors),
        month_matrix=np.stack(weights),
        months=np.array(months),
        midpoints=np.array([(a + pd.Timedelta(days=2.5)).isoformat() for a in starts]),
        interval_starts=np.asarray(starts[19:], dtype="datetime64[ns]"),
        interval_ends=np.asarray(ends[19:], dtype="datetime64[ns]"),
        origin=np.array(origin.isoformat()),
        start_lead=start,
        end_lead=stop,
        next_start_lead=next_start,
    )
