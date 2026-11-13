# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Exact-release, bounded-memory streaming of full planes into local crops."""

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fsspec
import numpy as np
import torch
import xarray as xr
import zarr  # type: ignore[import-untyped]

DEPTHS = np.array(
    [
        2.5,
        10,
        22.5,
        40,
        65,
        105,
        165,
        250,
        375,
        550,
        775,
        1050,
        1400,
        1850,
        2400,
        3100,
        4000,
        5000,
        6000,
    ],
    dtype=np.float32,
)
CHANNELS = [f"{field}_{z}" for field in ("uo", "vo", "thetao", "so") for z in range(19)]
CHANNELS += ["zos", "tauuo", "tauvo", "hfds"]
TEMPERATURES = [f"thetao_{z}" for z in range(19)]
SPLITS = {
    "train": ("1975-01-03", "2013-10-05"),
    "validation": ("2013-10-05", "2014-10-05"),
    "test": ("2014-10-10", "2022-12-24"),
}


class OceanSource:
    target_channels = tuple(range(38, 57))

    def __init__(self, root, readers=16, reference_root=None):
        self.root = root.rstrip("/")
        self.reference_root = (reference_root or root).rstrip("/")
        mapper = fsspec.get_mapper(self.root + "/OM4.zarr")
        self.metadata_sha256 = hashlib.sha256(mapper[".zmetadata"]).hexdigest()
        self.zarr = zarr.open_consolidated(mapper, mode="r")
        self.metadata = xr.open_zarr(mapper, consolidated=True, chunks=None)
        means = xr.open_zarr(
            fsspec.get_mapper(self.reference_root + "/OM4_means.zarr"), chunks=None
        )
        stds = xr.open_zarr(
            fsspec.get_mapper(self.reference_root + "/OM4_stds.zarr"), chunks=None
        )
        self.mean = np.array(
            [float(means[n].values) for n in CHANNELS], dtype=np.float32
        )
        self.std = np.array([float(stds[n].values) for n in CHANNELS], dtype=np.float32)
        if (
            not np.isfinite(self.mean).all()
            or not np.isfinite(self.std).all()
            or (self.std <= 0).any()
        ):
            raise ValueError("Invalid source normalization")
        self.pool = ThreadPoolExecutor(max_workers=readers)
        self.wet = np.stack(
            list(
                self.pool.map(
                    lambda z: self.zarr[f"mask_{z}"][:].astype(bool), range(19)
                )
            )
        )
        self.channel_wet = np.concatenate([self.wet] * 4 + [self.wet[:1]] * 4)
        self.lat = np.asarray(self.zarr["y"][:], dtype=np.float64)
        self.lon = np.asarray(self.zarr["x"][:], dtype=np.float64)
        self.height, self.width = len(self.lat), len(self.lon)
        if self.wet.shape != (19, self.height, self.width):
            raise ValueError("Mask/grid shape mismatch")
        lat, lon = np.meshgrid(
            np.deg2rad(self.lat), np.deg2rad(self.lon), indexing="ij"
        )
        self.xyz = np.stack(
            (np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)), -1
        ).astype(np.float32)
        reference = self.zarr
        self.reference_checks: dict[str, object] = {}
        if self.reference_root != self.root:
            ref_map = fsspec.get_mapper(self.reference_root + "/OM4.zarr")
            reference = zarr.open_consolidated(ref_map, mode="r")
            self.reference_checks["metadata_sha256"] = hashlib.sha256(
                ref_map[".zmetadata"]
            ).hexdigest()
            for name in ("x", "y", "time"):
                np.testing.assert_allclose(
                    self.zarr[name][:], reference[name][:], rtol=0, atol=1e-10
                )
            if dict(self.zarr["time"].attrs) != dict(reference["time"].attrs):
                raise ValueError("Local/reference time metadata differ")
            for z in range(19):
                np.testing.assert_array_equal(
                    self.wet[z], reference[f"mask_{z}"][:].astype(bool)
                )
            differences = {}
            for suffix, expected in (("means", self.mean), ("stds", self.std)):
                local = xr.open_zarr(
                    fsspec.get_mapper(self.root + f"/OM4_{suffix}.zarr"), chunks=None
                )
                actual = np.array(
                    [float(local[n].values) for n in CHANNELS], dtype=np.float32
                )
                differences[suffix] = float(np.max(np.abs(actual - expected)))
                local.close()
            self.reference_checks.update(
                coordinates_times_masks_match=True,
                local_normalization_max_abs_difference=differences,
                normalization_used="reference release",
            )
        self.area = np.asarray(reference["areacello"][:], dtype=np.float64)
        if self.area.shape != (self.height, self.width):
            raise ValueError("Expected native horizontal cell areas")
        self.locations, self.probabilities, self.location_cdfs = [], [], []
        for wet in self.wet:
            indices = np.flatnonzero(wet)
            weights = self.area.ravel()[indices]
            if (
                not len(indices)
                or not np.isfinite(weights).all()
                or (weights <= 0).any()
            ):
                raise ValueError("Invalid wet-cell area weights")
            self.locations.append(indices)
            self.probabilities.append(weights / weights.sum())
            cdf = np.cumsum(self.probabilities[-1])
            cdf /= cdf[-1]
            self.location_cdfs.append(cdf)
        dates = self.metadata.time.values
        self.date_strings = np.array([str(t)[:10] for t in dates])
        deltas = np.array(
            [
                (dates[i + 1] - dates[i]).total_seconds() / 86400
                for i in range(len(dates) - 1)
            ]
        )
        self.valid_transitions = deltas == 5
        self.origins = {}
        for split, (start, stop) in SPLITS.items():
            # Require both history records and target within the declared split.
            valid = (self.date_strings[:-2] >= start) & (self.date_strings[2:] <= stop)
            valid &= (deltas[:-1] == 5) & (deltas[1:] == 5)
            self.origins[split] = np.flatnonzero(valid) + 1
            if not len(self.origins[split]):
                raise ValueError(f"No valid five-day windows in {split}")

    def close(self):
        self.pool.shutdown(wait=True)
        self.metadata.close()

    def read(self, origin, with_target=True):
        if origin < 1 or origin >= len(self.date_strings) - int(with_target):
            raise ValueError("Origin lacks the requested history/target")
        if not self.valid_transitions[origin - 1] or (
            with_target and not self.valid_transitions[origin]
        ):
            raise ValueError("Window violates the declared five-day cadence")
        start = time.monotonic()
        requests = [
            (t, i, n) for t in (origin - 1, origin) for i, n in enumerate(CHANNELS)
        ]
        if with_target:
            requests += [(origin + 1, i, CHANNELS[i]) for i in self.target_channels]

        def read_plane(request):
            t, i, n = request
            a = np.asarray(self.zarr[n][int(t)], dtype=np.float32)
            wet = self.channel_wet[i]
            if not np.isfinite(a[wet]).all():
                raise ValueError(f"Non-finite wet data: {n} at {t}")
            return np.where(wet, (a - self.mean[i]) / self.std[i], 0).astype(np.float32)

        planes = list(self.pool.map(read_plane, requests))
        return {
            "input": np.stack(planes[:160]).reshape(2, 80, self.height, self.width),
            "target": np.stack(planes[160:]) if with_target else None,
            "origin": int(origin),
            "read_seconds": time.monotonic() - start,
        }

    def sample(self, frame, rng, size, crop, depth_sequence=None, points=None):
        if crop < 3 or crop % 2 != 1:
            raise ValueError("Crop must have odd side length >= 3")
        depths = (
            rng.integers(0, 19, size)
            if depth_sequence is None
            else np.asarray(depth_sequence)
        )
        if len(depths) != size:
            raise ValueError("Depth sequence length mismatch")
        if points is not None and len(points) != size:
            raise ValueError("Point sequence length mismatch")
        samples = []
        offsets = np.arange(crop) - crop // 2
        for i, depth in enumerate(depths):
            if not 0 <= depth < 19:
                raise ValueError("Invalid target depth")
            if points is None:
                # Reuse the area CDF instead of rebuilding a million-cell table
                # for every crop. One uniform draw preserves the sampling law.
                index = np.searchsorted(
                    self.location_cdfs[depth], rng.random(), side="right"
                )
                flat = self.locations[depth][index]
                row, col = divmod(int(flat), self.width)
            else:
                row, col = points[i]
                if not (
                    0 <= row < self.height
                    and 0 <= col < self.width
                    and self.wet[depth, row, col]
                ):
                    raise ValueError("Query must be a wet cell at its target depth")
            raw_rows = row + offsets
            rows = raw_rows.clip(0, self.height - 1)
            cols = (col + offsets) % self.width
            inside = np.broadcast_to(
                ((raw_rows >= 0) & (raw_rows < self.height))[:, None], (crop, crop)
            )
            wet = (
                self.wet[:, rows[:, None], cols].transpose(1, 2, 0) & inside[..., None]
            )
            values = (
                frame["input"][:, :, rows[:, None], cols].transpose(0, 2, 3, 1).copy()
            )
            values[:, ~inside] = 0
            xyz = self.xyz[rows[:, None], cols]
            geometry = np.concatenate((xyz, xyz - self.xyz[row, col]), -1)
            samples.append(
                {
                    "values": values.reshape(2 * crop * crop, 80),
                    "wet": np.tile(wet.reshape(-1, 19), (2, 1)).astype(np.float32),
                    "geometry": np.tile(geometry.reshape(-1, 6), (2, 1)),
                    "valid": np.tile(wet.any(-1).ravel(), 2),
                    "depth": np.int64(depth),
                    "depth_m": DEPTHS[depth],
                    "current": frame["input"][1, 38 + depth, row, col],
                    "target": frame["target"][depth, row, col]
                    if frame["target"] is not None
                    else np.float32(np.nan),
                    "row": np.int64(row),
                    "col": np.int64(col),
                    "origin": np.int64(frame["origin"]),
                }
            )
        return {
            key: torch.from_numpy(np.stack([sample[key] for sample in samples]))
            for key in samples[0]
        }

    def receipt(self):
        return {
            "root": self.root,
            "reference_root": self.reference_root,
            "reference_checks": self.reference_checks,
            "metadata_sha256": self.metadata_sha256,
            "channels": CHANNELS,
            "depths_m": DEPTHS.tolist(),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "splits": {
                name: {"origins": v.tolist(), "count": len(v)}
                for name, v in self.origins.items()
            },
            "cadence_days": 5,
            "input_offsets": [-1, 0],
            "target_offset": 1,
            "sampling": "uniform depth, cell-area proportional among wet cells at that depth",
            "data_created": self.zarr.attrs.get("m2lines/date_created"),
        }


def audit_support(source, destination, max_dates=None):
    """Training-only range scan, resumable by per-date receipts; no validation fit."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = source.receipt()
    dates = np.unique(
        np.concatenate(
            [
                source.origins["train"] - 1,
                source.origins["train"],
                source.origins["train"] + 1,
            ]
        )
    )
    if max_dates is not None:
        dates = dates[
            np.linspace(0, len(dates) - 1, min(max_dates, len(dates)), dtype=int)
        ]
    rows = []
    for date in dates:
        path = destination / f"range-{date:04d}.json"
        if path.exists():
            row = json.loads(path.read_text())
            if row["source_sha256"] != source.metadata_sha256:
                raise ValueError("Support audit source changed")
        else:

            def depth_limits(depth):
                a = np.asarray(source.zarr[TEMPERATURES[depth]][int(date)])[
                    source.wet[depth]
                ]
                if not np.isfinite(a).all():
                    raise ValueError(f"Nonfinite training temperature {date}/{depth}")
                a = (a - source.mean[38 + depth]) / source.std[38 + depth]
                return [float(a.min()), float(a.max())]

            row = {
                "date": int(date),
                "source_sha256": source.metadata_sha256,
                "limits": list(source.pool.map(depth_limits, range(19))),
            }
            path.write_text(json.dumps(row) + "\n")
        rows.append(row)
        if len(rows) % 25 == 0 or len(rows) == len(dates):
            print(
                json.dumps({"audit_dates": len(rows), "total": len(dates)}), flush=True
            )
    limits = np.array([r["limits"] for r in rows])
    absolute = np.abs(limits).max((0, 2)) * 1.01
    # The full training range bounds any difference between two training values.
    residual = (limits[:, :, 1].max(0) - limits[:, :, 0].min(0)) * 1.01
    result = {
        "source": manifest,
        "complete_training_scan": max_dates is None,
        "dates_scanned": len(dates),
        "absolute_bounds": absolute.tolist(),
        "residual_bounds": residual.tolist(),
    }
    path = destination / ("support.json" if max_dates is None else "pilot-support.json")
    path.write_text(json.dumps(result, indent=2) + "\n")
    return path
