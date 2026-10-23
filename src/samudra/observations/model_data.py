# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Native-calendar OM4/CM4 windows on the verified common Gaussian grid."""

import hashlib
from datetime import timedelta
from pathlib import Path

import cftime
import numpy as np
import torch
import zarr  # type: ignore[import-untyped]

from samudra.constants import build_om4_layout
from samudra.observations.intervals import calendar_weights


def calendar_context(lat, lon, dates):
    latitude, longitude = np.meshgrid(np.deg2rad(lat), np.deg2rad(lon), indexing="ij")
    xyz = np.stack(
        [
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        ]
    )
    result = []
    for stamp in dates:
        first = type(stamp)(stamp.year, 1, 1)
        last = type(stamp)(stamp.year + 1, 1, 1)
        phase = (
            2 * np.pi * (stamp - first).total_seconds() / (last - first).total_seconds()
        )
        season = np.broadcast_to(
            np.array([np.sin(phase), np.cos(phase)])[:, None, None],
            (2, *latitude.shape),
        )
        result.append(np.concatenate((xyz, season)))
    return np.asarray(result, dtype=np.float32)


class ModelMonths:
    """Read physical fields using shared observation scales and native forcing/masks.

    OM4 uses the established pre-validation cutoff. CM4 uses its first 180 model
    years for training, reserving the final twenty for future diagnostics.
    """

    def __init__(self, path, kind, reference, om4_statistics):
        if kind not in ("om4", "cm4"):
            raise ValueError(kind)
        self.kind = kind
        self.source_metadata_sha256 = hashlib.sha256(
            (Path(path) / ".zmetadata").read_bytes()
        ).hexdigest()
        self.store = zarr.open_consolidated(str(path), mode="r")
        self.grid = reference
        self.names = reference["names"].tolist()
        if self.names != [
            f"{v}_{i}" for v in ("uo", "vo", "thetao", "so") for i in range(19)
        ] + ["zos"]:
            raise ValueError("Unexpected physical channel order")
        z = self.store
        lat = z["lat"][:, 0] if kind == "cm4" else z["y"][:]
        lon = z["lon"][0] if kind == "cm4" else z["x"][:]
        np.testing.assert_allclose(lat, reference["lat"], atol=1e-6, rtol=0)
        np.testing.assert_allclose(lon, reference["lon"], atol=1e-6, rtol=0)
        self.lat, self.lon = lat, lon
        if kind == "cm4":
            np.testing.assert_allclose(
                z["lat"][:],
                np.broadcast_to(lat[:, None], (len(lat), len(lon))),
                atol=1e-6,
                rtol=0,
            )
            np.testing.assert_allclose(
                z["lon"][:],
                np.broadcast_to(lon[None], (len(lat), len(lon))),
                atol=1e-6,
                rtol=0,
            )
            np.testing.assert_allclose(
                z["dz"][:], build_om4_layout().depth_thickness, rtol=0, atol=1e-6
            )
            wet = z["wetmask"][:].astype(bool)
            self.mask = np.concatenate([wet] * 4 + [wet[:1]])
            bounds = z["time_bnds"]
            self.bounds = cftime.num2date(
                bounds[:], bounds.attrs["units"], bounds.attrs["calendar"]
            )
            labels = cftime.num2date(
                z["time"][:], z["time"].attrs["units"], z["time"].attrs["calendar"]
            )
            if not np.array_equal(labels, self.bounds[:, 1]):
                raise ValueError("CM4 timestamps are not verified interval ends")
        else:
            self.mask = np.stack(
                [z[f"mask_{i}"][:] for i in range(19)] * 4 + [z["mask_0"][:]]
            ).astype(bool)
            labels = cftime.num2date(
                z["time"][:],
                z["time"].attrs["units"],
                z["time"].attrs.get("calendar", "standard"),
            )
            # Published OM4 five-day means are centered at 12:00 on day three.
            self.bounds = np.array(
                [[t - timedelta(days=2.5), t + timedelta(days=2.5)] for t in labels],
                dtype=object,
            )
        lengths = [(b - a).total_seconds() for a, b in self.bounds]
        if any(v != 5 * 86400 for v in lengths):
            raise ValueError("Non-five-day source intervals")
        gaps = self.bounds[1:, 0] != self.bounds[:-1, 1]
        if kind == "cm4" and gaps.any():
            raise ValueError("CM4 interval bounds are not contiguous")
        self.midpoints = [a + (b - a) / 2 for a, b in self.bounds]
        statistics = Path(om4_statistics)
        mean = zarr.open_consolidated(str(statistics / "OM4_means.zarr"), mode="r")
        std = zarr.open_consolidated(str(statistics / "OM4_stds.zarr"), mode="r")
        self.forcing_mean = np.array(
            [float(mean[n][...]) for n in ("tauuo", "tauvo", "hfds")], dtype=np.float32
        )
        self.forcing_std = np.array(
            [float(std[n][...]) for n in ("tauuo", "tauvo", "hfds")], dtype=np.float32
        )
        if not np.isfinite(self.forcing_mean).all() or not (self.forcing_std > 0).all():
            raise ValueError("Invalid forcing normalization")
        self.windows = []
        self.window_months = []
        self.excluded_gap_months = []
        first_year = self.bounds[0, 0].year
        for year in range(first_year, self.bounds[-1, 1].year + 1):
            for month in range(1, 13):
                first = type(self.midpoints[0])(year, month, 1)
                last = type(first)(year + (month == 12), month % 12 + 1, 1)
                indices = np.arange(
                    np.searchsorted(self.bounds[:, 1], first, side="right"),
                    np.searchsorted(self.bounds[:, 0], last, side="left"),
                )
                if (
                    not len(indices)
                    or indices[0] < 19
                    or self.bounds[indices[-1], 1] < last
                ):
                    continue
                if kind == "om4" and (
                    year < 1975
                    or (year, month) > (2013, 7)
                    or self.bounds[indices[0] - 19, 0].year < 1975
                ):
                    continue
                if kind == "cm4" and year >= first_year + 180:
                    continue
                # History frames may straddle missing days. Every forecast
                # transition, including the last history frame -> first target,
                # must still span exactly one contiguous five-day interval.
                # TODO: add temporal encoding and random history dropout during
                # training to improve initializer robustness to missing days.
                if gaps[indices[0] - 1 : indices[-1]].any():
                    self.excluded_gap_months.append(str(first))
                    continue
                weights = calendar_weights(
                    self.bounds[indices, 0], self.bounds[indices, 1], first, last
                )
                self.window_months.append(first)
                self.windows.append(
                    (indices[0], indices[-1] + 1, weights, (last - first).days)
                )
        if not self.windows:
            raise ValueError("No training months")

    def sample(self, index, device):
        start, stop, weights, duration = self.windows[index % len(self.windows)]
        return self._read(
            start,
            stop,
            weights,
            duration,
            device,
            origin_start=start,
            next_start=self._next_start(self.window_months[index % len(self.windows)]),
        )

    def _next_start(self, first):
        last = type(first)(first.year + (first.month == 12), first.month % 12 + 1, 1)
        return int(np.searchsorted(self.bounds[:, 0], last, side="right") - 1)

    def _continuation_window(self, index, month_offset):
        if month_offset < 0 or month_offset >= 12:
            raise ValueError("Replay month offset must be in 0..11")
        origin_start = self.windows[index][0]
        original = self.window_months[index]
        month_id = original.year * 12 + original.month - 1 + month_offset
        first = type(original)(month_id // 12, month_id % 12 + 1, 1)
        candidates = [i for i, month in enumerate(self.window_months) if month == first]
        if not candidates:
            return None
        start, stop, weights, duration = self.windows[candidates[0]]
        if any(
            self.bounds[k, 1] != self.bounds[k + 1, 0]
            # Check from the original forecast boundary, not its input history.
            for k in range(origin_start - 1, stop - 1)
        ):
            return None
        if stop - origin_start > 73:
            return None
        return start, stop, weights, duration, origin_start, self._next_start(first)

    def continuation_layout(self, index, month_offset):
        """Inspect eligibility without reading any physical field data."""
        window = self._continuation_window(index, month_offset)
        if window is None:
            return None
        start, stop, _, _, origin_start, next_start = window
        return dict(
            start_lead=int(start - origin_start),
            end_lead=int(stop - origin_start),
            next_start_lead=int(next_start - origin_start),
        )

    def continuation(self, index, month_offset, device):
        window = self._continuation_window(index, month_offset)
        if window is None:
            raise ValueError(
                "Replay crosses a source gap, training cutoff, or one-year limit"
            )
        start, stop, weights, duration, origin_start, next_start = window
        return self._read(
            start,
            stop,
            weights,
            duration,
            device,
            origin_start=origin_start,
            next_start=next_start,
        )

    def _read(
        self, start, stop, weights, duration, device, *, origin_start, next_start
    ):
        interval = slice(start - 19, stop)
        z = self.store
        if self.kind == "cm4":
            values = np.concatenate(
                [z[v][interval] for v in ("uo", "vo", "thetao", "so")]
                + [z["zos"][interval][:, None]],
                axis=1,
            )
        else:
            values = np.stack([z[n][interval] for n in self.names], axis=1)
        if not np.isfinite(values[:, self.mask]).all():
            raise ValueError("Nonfinite model target on declared wet support")
        fields = np.where(
            self.mask,
            (values - self.grid["mean"][:, None, None])
            / self.grid["std"][:, None, None],
            0,
        ).astype(np.float32)
        forcing = np.stack([z[n][interval] for n in ("tauuo", "tauvo", "hfds")], axis=1)
        if not np.isfinite(forcing[:, :, self.mask[38]]).all():
            raise ValueError("Nonfinite own-model forcing on wet support")
        forcing = np.where(
            self.mask[38],
            (forcing - self.forcing_mean[:, None, None])
            / self.forcing_std[:, None, None],
            0,
        ).astype(np.float32)

        def tensor(value):
            return torch.as_tensor(value, device=device)

        return dict(
            fields=tensor(fields)[None],
            forcing=tensor(forcing)[None],
            contexts=tensor(
                calendar_context(self.lat, self.lon, self.midpoints[start - 19 : stop])
            )[None],
            mask=tensor(self.mask),
            weights=tensor(np.asarray(weights, dtype=np.float32)),
            duration=int(duration),
            first=str(self.bounds[start, 0]),
            start_lead=int(start - origin_start),
            end_lead=int(stop - origin_start),
            next_start_lead=int(next_start - origin_start),
            origin_index=int(origin_start),
        )
