# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Observation-fitted state scales, bounded CPU samples, and explicit replay windows."""

import json
from functools import cached_property, lru_cache
from pathlib import Path

import numpy as np
import torch

from samudra.observations.archive import context_planes, validate_normalization_policy


class Samples:
    def __init__(self, root, device, *, daily=None, normalization=None):
        self.root = Path(root)
        self.device = device
        self.daily = Path(daily) if daily is not None else self.root / "daily"
        self.normalization = (
            Path(normalization) if normalization is not None else self.root
        )
        validate_normalization_policy(
            json.loads((self.normalization / "normalization.json").read_text())
        )
        self.grid = dict(np.load(self.root / "grid.npz"))
        self.stats = dict(np.load(self.normalization / "statistics.npz"))
        self.mask = self.tensor(self.grid["mask"])
        self.area = self.tensor(np.cos(np.deg2rad(self.grid["lat"]))[:, None])
        self.ts_indices = list(range(38, 52)) + list(range(57, 71))
        self.ts_mask = self.mask[self.ts_indices].clone()
        self.ts_mask[0] = 0  # Supplied surface temperature is not interior supervision.
        self.ts_scale = self.tensor(self.stats["interior_std"])[None, :, None, None]
        self.surface_scale = self.tensor(self.stats["surface_std"])[
            None, None, :, None, None
        ]
        with np.load(self.normalization / "normalization.npz") as normalization:
            self.grid["mean"] = normalization["state_mean"]
            self.grid["std"] = normalization["state_std"]
        if (
            self.grid["mean"].shape != (77,)
            or self.grid["std"].shape != (77,)
            or not np.isfinite(self.grid["mean"]).all()
            or not np.isfinite(self.grid["std"]).all()
            or not (self.grid["std"] > 0).all()
        ):
            raise ValueError("Invalid observation normalization artifact")
        self.mean = self.tensor(self.grid["mean"])[None, None, :, None, None]
        self.std = self.tensor(self.grid["std"])[None, None, :, None, None]

    def tensor(self, array):
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    def paths(self, split):
        return sorted((self.root / split).glob("*.npz"))

    @cached_property
    def read_sample(self):
        # Keep only four CPU months; the full archive never resides on GPU.
        @lru_cache(maxsize=4)
        def read(path):
            with np.load(path) as saved:
                return {k: saved[k] for k in saved.files}

        return read

    def load(self, path):
        sample = self.read_sample(Path(path))
        return self.normalize(sample, Path(path).stem)

    def normalize(self, sample, name):
        raw = sample["surface"]
        validity = np.isfinite(raw) & self.grid["mask"][[38, 76]]
        filled = np.where(validity, raw, 0)
        normalized = (filled - self.grid["mean"][[38, 76], None, None]) / self.grid[
            "std"
        ][[38, 76], None, None]
        normalized = np.where(validity, normalized, 0)
        normalized *= self.grid["mask"][[38, 76]]
        atmosphere = (
            sample["atmosphere"] - self.stats["atmosphere_mean"][None, :, None, None]
        ) / self.stats["atmosphere_std"][None, :, None, None]
        if not np.isfinite(normalized).all() or not np.isfinite(atmosphere).all():
            raise ValueError(f"Nonfinite inputs: {name}")
        interior = sample["interior"].reshape(28, *raw.shape[-2:])
        return dict(
            surface=self.tensor(normalized)[None],
            atmosphere=self.tensor(atmosphere)[None],
            contexts=self.tensor(
                context_planes(self.grid["lat"], self.grid["lon"], sample["midpoints"])
            )[None],
            validity=self.tensor(validity)[None],
            month_weights=self.tensor(sample["month_weights"]),
            interior=self.tensor(interior)[None],
            raw_surface=self.tensor(raw[19:])[None],
            raw=sample,
            name=name,
        )

    def trajectory(self, origin, *, target_month=None, steps=None, split="train"):
        from samudra.observations.archive import ObservationArchive, trajectory

        raw = trajectory(
            ObservationArchive(self.daily),
            origin,
            target_month=target_month,
            steps=steps,
            split=split,
        )
        result = self.normalize(raw, str(target_month or origin)[:7])
        result.update(
            monthly_targets=self.tensor(
                raw["monthly_targets"].reshape(
                    len(raw["months"]), 28, *self.grid["mask"].shape[-2:]
                )
            )[None],
            month_matrix=self.tensor(raw["month_matrix"]),
            months=raw["months"].tolist(),
            interval_starts=raw["interval_starts"],
            interval_ends=raw["interval_ends"],
            start_lead=raw["start_lead"],
            end_lead=raw["end_lead"],
            next_start_lead=raw["next_start_lead"],
        )
        return result

    def physical(self, normalized):
        return normalized.float() * self.std + self.mean

    def masked_channel_mse(self, prediction, target, mask, scale):
        valid = torch.isfinite(target) & mask.bool()
        weight = valid * self.area
        safe_target = torch.where(valid, target, 0)
        safe_prediction = torch.where(valid, prediction.float(), 0)
        denominator = weight.sum((-2, -1))
        error = ((safe_prediction - safe_target) / scale).square()
        reduced = (error * weight).sum((-2, -1)) / denominator.clamp_min(1e-12)
        present = denominator > 0
        if not present.any():
            raise ValueError("Empty loss support")
        return reduced, present

    def interior_loss(self, monthly_normalized, sample):
        physical = self.physical(monthly_normalized[:, None])[:, 0, self.ts_indices]
        channel, present = self.masked_channel_mse(
            physical, sample["interior"], self.ts_mask, self.ts_scale
        )
        groups = []
        for region in (slice(0, 14), slice(14, 28)):
            values, valid = channel[:, region], present[:, region]
            if not valid.any():
                raise ValueError("Missing thermohaline group")
            groups.append(values[valid].mean())
        return (groups[0] + groups[1]) / 2

    def forecast_losses(self, prediction, sample):
        weights = sample["month_weights"]
        monthly = (prediction * weights[None, :, None, None, None]).sum(1)
        interior = self.interior_loss(monthly, sample)
        surface = self.physical(prediction)[:, :, [38, 76]]
        errors, present = self.masked_channel_mse(
            surface, sample["raw_surface"], self.mask[[38, 76]], self.surface_scale
        )
        groups = []
        for channel in range(2):
            values, valid = errors[:, :, channel], present[:, :, channel]
            if not valid.any():
                raise ValueError("Missing forecast surface group")
            groups.append(values[valid].mean())
        return {"monthly_ts": interior, "sst": groups[0], "ssh": groups[1]}
