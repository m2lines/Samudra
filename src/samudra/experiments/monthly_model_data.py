# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Native-calendar OM4/CM4 windows on the verified common Gaussian grid."""

import hashlib
import json
from datetime import timedelta
from pathlib import Path

import cftime
import numpy as np
import torch
import zarr  # type: ignore[import-untyped]

from samudra.constants import build_om4_layout


def calendar_weights(starts, ends, first, last):
    """Exact interval intersections; works with datetime and cftime calendars."""
    intervals = list(zip(starts, ends, strict=True))
    cursor = first
    for a, b in intervals:
        if b <= a:
            raise ValueError("Invalid interval bounds")
        lo, hi = max(a, first), min(b, last)
        if hi > lo:
            if lo != cursor:
                raise ValueError("Intervals overlap or leave a gap")
            cursor = hi
    if cursor != last:
        raise ValueError("Intervals do not exactly cover requested month")
    spans = [
        max(0.0, (min(b, last) - max(a, first)).total_seconds()) for a, b in intervals
    ]
    duration = (last - first).total_seconds()
    if duration <= 0 or abs(sum(spans) - duration) > 1e-6:
        raise ValueError("Intervals do not exactly cover requested month")
    return [s / duration for s in spans]


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
    """Read physical fields using fixed OM4 scales, own forcing and own wet mask.

    OM4 uses the established pre-validation cutoff. CM4 uses its first 180 model
    years for training, reserving the final twenty for future diagnostics.
    """

    def __init__(self, path, kind, reference, om4_statistics, packed=None):
        if kind not in ("om4", "cm4"):
            raise ValueError(kind)
        self.kind = kind
        self.store = zarr.open_consolidated(str(path), mode="r")
        self.grid = reference
        self.names = reference["names"].tolist()
        if self.names != [
            f"{v}_{i}" for v in ("uo", "vo", "thetao", "so") for i in range(19)
        ] + ["zos"]:
            raise ValueError("Unexpected physical channel order")
        self.packed = None
        if packed is not None:
            packed = Path(packed)
            proof = json.loads((packed / "COMPLETE.json").read_text())
            if (
                proof["source"] != str(path)
                or proof["source_metadata_sha256"]
                != hashlib.sha256((Path(path) / ".zmetadata").read_bytes()).hexdigest()
            ):
                raise ValueError("Packed source contract differs")
            if (
                proof["verification"]
                != "every packed frame read back and byte-hash matched to source"
            ):
                raise ValueError("Packed data lacks exact verification")
            self.packed = np.load(packed / "frames.npy", mmap_mode="r")
            if (
                list(self.packed.shape) != proof["shape"]
                or self.packed.dtype != np.float32
            ):
                raise ValueError("Packed shape/dtype differs")
            self.packed_indices = {
                original: offset for offset, original in enumerate(proof["indices"])
            }
            self.packed_hashes = proof["frame_sha256"]
            self.verified_frames = set()
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
                if gaps[indices[0] - 19 : indices[-1]].any():
                    self.excluded_gap_months.append(str(first))
                    continue
                weights = calendar_weights(
                    self.bounds[indices, 0], self.bounds[indices, 1], first, last
                )
                self.windows.append(
                    (indices[0], indices[-1] + 1, weights, (last - first).days)
                )
        if not self.windows:
            raise ValueError("No training months")

    def sample(self, index, device):
        start, stop, weights, duration = self.windows[index % len(self.windows)]
        interval = slice(start - 19, stop)
        z = self.store
        if self.packed is not None:
            positions = [self.packed_indices[i] for i in range(start - 19, stop)]
            for offset in positions:
                if offset not in self.verified_frames:
                    if (
                        hashlib.sha256(self.packed[offset].tobytes()).hexdigest()
                        != self.packed_hashes[offset]
                    ):
                        raise ValueError("Packed frame checksum changed")
                    self.verified_frames.add(offset)
            values = np.asarray(self.packed[positions])
        elif self.kind == "cm4":
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
            duration=duration,
            first=str(self.bounds[start, 0]),
        )
