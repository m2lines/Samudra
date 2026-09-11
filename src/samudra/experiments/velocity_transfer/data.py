# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from samudra.experiments.velocity_transfer.prepare import SPLITS

HISTORY = 4


def eligible_anchors(times: np.ndarray, split: str, steps: int = 6) -> np.ndarray:
    """Require the entire history/forecast window to lie within one time split."""
    start, end = (np.datetime64(s) for s in SPLITS[split])
    days = times.astype("datetime64[D]")
    candidates = np.arange(HISTORY - 1, len(times) - steps)
    candidates = candidates[
        (days[candidates - HISTORY + 1] >= start) & (days[candidates + steps] <= end)
    ]
    gaps = np.diff(times).astype("timedelta64[h]").astype(float) / 24
    return np.array(
        [
            i
            for i in candidates
            if (
                (gaps[i - HISTORY + 1 : i + steps] >= 4)
                & (gaps[i - HISTORY + 1 : i + steps] <= 6)
            ).all()
        ],
        dtype=np.int64,
    )


def geometry(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Spherical position, physical cell spacing, and spherical cell area."""
    lat, lon = np.meshgrid(np.deg2rad(y), np.deg2rad(x), indexing="ij")
    dy = np.gradient(np.deg2rad(y))[:, None] * 6_371_000
    dx = np.gradient(np.unwrap(np.deg2rad(x)))[None, :] * 6_371_000 * np.cos(lat)
    sphere = [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    spacing = [
        np.log(np.maximum(dx, 1) / 100_000),
        np.broadcast_to(np.log(np.maximum(dy, 1) / 100_000), lat.shape),
    ]
    return np.stack(sphere + spacing).astype(np.float32), (
        np.abs(dx * dy) / 1e10
    ).astype(np.float32)


class VelocitySource:
    """Read bounded float32 batches through the Rust flat-array reader."""

    def __init__(self, path: Path, pool):
        from samudra_rust_loader import FlatOm4Reader

        self.path = path
        manifest = json.loads((path / "manifest.json").read_text())
        if not manifest["complete"]:
            raise ValueError(f"Incomplete preparation: {path}")
        self.kind = manifest["kind"]
        self.reader = FlatOm4Reader(path / "fields.zarr", ["u", "v"], pool)
        with xr.open_zarr(path / "fields.zarr") as ds:
            self.times = ds.time.values.astype("datetime64[ns]")
            self.y, self.x = ds.y.values, ds.x.values
        with np.load(path / "stats.npz") as stats:
            self.mean = stats["mean"]
            self.std = stats["std"][:, None, None]
            self.mask = stats["mask"]
            self.monthly_mean = stats["monthly_mean"]
        coords, self.area = geometry(self.y, self.x)
        self.static = np.concatenate(
            (coords, self.mask[None], self.mean / self.std)
        ).astype(np.float32)

    def batch(
        self,
        anchor: int,
        steps: int,
        device: torch.device,
        crop: tuple[int, int, int, int] | None = None,
    ):
        indexes = list(range(anchor - HISTORY + 1, anchor + steps + 1))
        data = np.empty((len(indexes), 2, *self.mask.shape), dtype=np.float32)
        self.reader.read_into(indexes, ["u", "v"], data)
        fields = {
            "values": data,
            "static": self.static,
            "area": self.area[None],
            "mask": self.mask[None],
            "mean": self.mean,
            "std": np.broadcast_to(self.std, self.mean.shape),
        }
        if crop is not None:
            y, x, size, halo = crop
            if not (
                halo * 2 < size <= self.mask.shape[0]
                and size <= self.mask.shape[1]
                and 0 <= y <= self.mask.shape[0] - size
            ):
                raise ValueError(f"Invalid regional crop: {crop}")
            for key, value in fields.items():
                fields[key] = np.take(
                    value[..., y : y + size, :],
                    np.arange(x, x + size) % self.mask.shape[1],
                    axis=-1,
                )
        valid = (
            np.isfinite(fields["values"]).all(axis=1, keepdims=True)
            & fields["mask"][None]
        )
        normalized = (fields.pop("values") - fields["mean"]) / fields["std"]
        normalized = np.where(valid, normalized, 0)
        fields["history"] = normalized[:HISTORY]
        fields["targets"] = normalized[HISTORY:]
        fields["history_valid"] = valid[:HISTORY]
        fields["target_valid"] = valid[HISTORY:] & valid[:HISTORY].all(axis=0)
        if crop is not None:
            halo = crop[-1]
            interior = np.zeros_like(fields["mask"])
            interior[..., halo:-halo, halo:-halo] = 1
            fields["target_valid"] &= interior
        times = self.times[indexes]
        fields["day_offsets"] = (
            (times - times[HISTORY - 1]) / np.timedelta64(1, "D")
        ).astype(np.float32)
        fields["year_day"] = np.array(
            (
                times[HISTORY - 1].astype("datetime64[D]")
                - times[HISTORY - 1].astype("datetime64[Y]")
            )
            / np.timedelta64(1, "D"),
            dtype=np.float32,
        )
        result = {}
        for name, value in fields.items():
            tensor = torch.from_numpy(np.array(value, copy=True)).unsqueeze(0)
            if device.type == "cuda":
                tensor = tensor.pin_memory()
            result[name] = tensor.to(device, non_blocking=device.type == "cuda")
        return result

    def random_crop(self, rng: np.random.Generator, size: int = 384, halo: int = 128):
        """Choose regional interiors with usable ocean coverage, wrapping only at the real date line."""
        h, w = self.mask.shape
        if size > min(h, w):
            raise ValueError(f"Crop size {size} exceeds grid {h}x{w}")
        for _ in range(100):
            y, x = int(rng.integers(0, h - size + 1)), int(rng.integers(0, w))
            interior = np.take(
                self.mask[y + halo : y + size - halo],
                np.arange(x + halo, x + size - halo) % w,
                axis=-1,
            )
            if interior.mean() >= 0.25:
                return y, x, size, halo
        raise RuntimeError("Could not sample an ocean crop in 100 attempts")
