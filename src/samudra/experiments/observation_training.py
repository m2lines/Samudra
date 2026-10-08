# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Observation-only supervision and inference inputs for transferred D."""

from functools import cached_property, lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

from samudra.constants import build_om4_layout
from samudra.experiments.observation_data import context_planes
from samudra.metrics import kernels


class Samples:
    def __init__(
        self, root, device, surface_fill="climatology", global_observations=False
    ):
        if surface_fill not in ("climatology", "zero"):
            raise ValueError("Unknown missing surface input fill")
        self.surface_fill = surface_fill
        self.root = Path(root)
        self.device = device
        self.grid = dict(np.load(self.root / "grid.npz"))
        self.stats = dict(np.load(self.root / "statistics.npz"))
        self.mask = self.tensor(self.grid["mask"])
        self.mean = self.tensor(self.grid["mean"])[None, None, :, None, None]
        self.std = self.tensor(self.grid["std"])[None, None, :, None, None]
        self.area = self.tensor(np.cos(np.deg2rad(self.grid["lat"]))[:, None])
        self.global_observations = global_observations
        if not global_observations:
            self.area *= self.tensor(np.abs(self.grid["lat"][:, None]) <= 60)
        self.ts_indices = list(range(38, 52)) + list(range(57, 71))
        self.ts_mask = self.mask[self.ts_indices].clone()
        self.ts_mask[0] = 0  # Supplied surface temperature is not interior supervision.
        self.ts_scale = self.tensor(self.stats["interior_std"])[None, :, None, None]
        self.surface_scale = self.tensor(self.stats["surface_std"])[
            None, None, :, None, None
        ]

    def use_observation_normalization(self):
        """Scratch control: no simulation-derived normalization statistics."""
        mean, std = np.zeros(77, dtype=np.float32), np.ones(77, dtype=np.float32)
        for offset, group in ((38, 0), (57, 1)):
            mean[offset : offset + 14] = self.stats["interior_mean"][
                group * 14 : (group + 1) * 14
            ]
            std[offset : offset + 14] = self.stats["interior_std"][
                group * 14 : (group + 1) * 14
            ]
            # Unsupervised deep slots retain the deepest observed scale. No
            # simulation values or invented deep labels enter this control.
            mean[offset + 14 : offset + 19] = mean[offset + 13]
            std[offset + 14 : offset + 19] = std[offset + 13]
        mean[[38, 76]], std[[38, 76]] = (
            self.stats["surface_mean"],
            self.stats["surface_std"],
        )
        if "prepared_sample" in self.__dict__:
            self.prepared_sample.cache_clear()
        self.grid["mean"], self.grid["std"] = mean, std
        self.mean = self.tensor(mean)[None, None, :, None, None]
        self.std = self.tensor(std)[None, None, :, None, None]

    @cached_property
    def interior_climatology(self):
        h, w = self.grid["mask"].shape[-2:]
        total = np.zeros((12, 28, h, w), dtype=np.float64)
        count = np.zeros_like(total)
        for path in self.paths("train"):
            with np.load(path) as saved:
                values = saved["interior"].reshape(28, h, w)
            month = pd.Timestamp(path.stem).month - 1
            valid = np.isfinite(values)
            total[month] += np.where(valid, values, 0)
            count[month] += valid
        result = np.divide(
            total, count, out=np.full_like(total, np.nan), where=count > 0
        )
        fallback = self.stats["interior_mean"][None, :, None, None]
        return np.where(np.isfinite(result), result, fallback).astype(np.float32)

    def persistence_anomaly(self, initial, sample):
        state = initial[:, -1].clone()
        previous_month = pd.Timestamp(str(sample["raw"]["midpoints"][18])).month - 1
        target_month = pd.Timestamp(sample["name"]).month - 1
        difference = (
            self.interior_climatology[target_month]
            - self.interior_climatology[previous_month]
        )
        difference[0] = 0  # Preserve the same observed-surface persistence control.
        scale = self.std[0, 0, self.ts_indices]
        state[:, self.ts_indices] += self.tensor(difference)[None] / scale
        return state

    def climatology_prediction(self, sample):
        count = len(sample["month_weights"])
        state = torch.zeros(
            (1, count, 77, *self.grid["mask"].shape[-2:]), device=self.device
        )
        target_month = pd.Timestamp(sample["name"]).month - 1
        interior = self.tensor(self.interior_climatology[target_month])
        state[:, :, self.ts_indices] = (
            interior - self.mean[0, 0, self.ts_indices]
        ) / self.std[0, 0, self.ts_indices]
        months = pd.DatetimeIndex(sample["raw"]["midpoints"][19:]).month.to_numpy() - 1
        surface = self.tensor(self.stats["surface_climatology"][months])
        state[:, :, [38, 76]] = (surface - self.mean[0, 0, [38, 76]]) / self.std[
            0, 0, [38, 76]
        ]
        return state * self.mask

    def tensor(self, array):
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    def paths(self, split):
        return sorted((self.root / split).glob("*.npz"))

    @cached_property
    def read_sample(self):
        # Cache only CPU source arrays: normalization and GPU transfer stay on
        # the original path. 256 entries cover 243 training + 9 validation
        # months (20.5 GiB); historical evaluation evicts training entries.
        @lru_cache(maxsize=256)
        def read(path):
            with np.load(path) as saved:
                return {k: saved[k] for k in saved.files}

        return read

    @cached_property
    def prepared_sample(self):
        # Opt-in CPU cache for the fixed training/validation inputs. Model
        # outputs and CUDA tensors are never retained here.
        return lru_cache(maxsize=256)(self.prepare_sample)

    @cached_property
    def coverage_sample(self):
        @lru_cache(maxsize=256)
        def read(path):
            raw = self.read_sample(Path(path))["surface"]
            return np.isfinite(raw) & self.grid["mask"][[38, 76]]

        return read

    def load_coverage(self, path):
        return self.tensor(self.coverage_sample(Path(path))[:19])[None]

    def load(self, path):
        prepare = (
            self.prepared_sample
            if getattr(self, "cache_prepared", False)
            else self.prepare_sample
        )
        prepared = prepare(Path(path))
        return {
            k: v if k in ("raw", "name") else self.tensor(v)
            for k, v in prepared.items()
        }

    def prepare_sample(self, path):
        sample = self.read_sample(Path(path))
        raw = sample["surface"]
        validity = np.isfinite(raw) & self.grid["mask"][[38, 76]]
        months = pd.DatetimeIndex(sample["midpoints"]).month.to_numpy() - 1
        filled = np.where(validity, raw, self.stats["surface_climatology"][months])
        normalized = (filled - self.grid["mean"][[38, 76], None, None]) / self.grid[
            "std"
        ][[38, 76], None, None]
        if getattr(self, "surface_fill", "climatology") == "zero":
            normalized = np.where(validity, normalized, 0)
        normalized *= self.grid["mask"][[38, 76]]
        atmosphere = (
            sample["atmosphere"] - self.stats["atmosphere_mean"][None, :, None, None]
        ) / self.stats["atmosphere_std"][None, :, None, None]
        if not np.isfinite(normalized).all() or not np.isfinite(atmosphere).all():
            raise ValueError(f"Nonfinite inputs: {path}")
        interior = sample["interior"].reshape(28, *raw.shape[-2:])
        return dict(
            surface=normalized[None],
            atmosphere=atmosphere[None],
            contexts=context_planes(
                self.grid["lat"], self.grid["lon"], sample["midpoints"]
            )[None],
            validity=validity[None],
            month_weights=sample["month_weights"],
            interior=interior[None],
            raw_surface=raw[None, 19:],
            raw=sample,
            name=Path(path).stem,
        )

    def physical(self, normalized):
        # Latent slots have no physical scale or direct observation target.
        return normalized[:, :, : self.std.shape[2]].float() * self.std + self.mean

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

    def forecast_loss(self, prediction, sample):
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
        return 0.8 * interior + 0.1 * groups[0] + 0.1 * groups[1]

    def ohc(self, state):
        """Monthly normalized state -> native-layer OHC, complete model columns only."""
        physical = self.physical(state[:, None])[:, 0].detach().cpu().numpy()
        layout = build_om4_layout()
        temp = physical[:, 38:57]
        if not np.isfinite(np.where(self.grid["mask"][38:57], temp, 0)).all():
            raise ValueError("Nonfinite temperature on fixed model wet support")
        temp = np.where(self.grid["mask"][38:57], temp, np.nan)
        field = xr.DataArray(
            temp,
            dims=("time", "depth", "lat", "lon"),
            coords={
                "depth": np.array(layout.depth_levels),
                "lat": self.grid["lat"],
                "lon": self.grid["lon"],
            },
        )
        dz = xr.DataArray(
            np.array(layout.depth_thickness),
            dims="depth",
            coords={"depth": np.array(layout.depth_levels)},
        )
        maps = kernels.ohc_per_area_layer_maps(field, native_dz=dz)
        values = []
        for layer in kernels.OHC_LAYERS:
            thickness = kernels.layer_overlap_thickness(
                field.depth, layer.min_depth, layer.max_depth, native_dz=dz
            )
            complete = field.where(thickness > 0).notnull().sum("depth") == int(
                (thickness > 0).sum()
            )
            values.append(maps[layer.label].where(complete).values)
        return np.stack(values, axis=1)
