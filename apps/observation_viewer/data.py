# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Coordinate-aware access and diagnostics for immutable viewer exports."""

import hashlib
import json
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np


def edges(coordinates, bounds=None):
    coordinates = np.asarray(coordinates)
    if (
        coordinates.ndim != 1
        or len(coordinates) < 2
        or not np.all(np.diff(coordinates) > 0)
    ):
        raise ValueError("Coordinates must be strictly increasing")
    midpoints = (coordinates[1:] + coordinates[:-1]) / 2
    outer = (2 * coordinates[0] - midpoints[0], 2 * coordinates[-1] - midpoints[-1])
    return np.r_[
        bounds[0] if bounds else outer[0], midpoints, bounds[1] if bounds else outer[1]
    ]


def paired_stats(prediction, reference, lat):
    """Cosine-weighted snapshot diagnostics on common finite support."""
    valid = np.isfinite(prediction) & np.isfinite(reference)
    weights = np.where(valid, np.cos(np.deg2rad(lat))[:, None], 0)
    total = weights.sum()
    if total == 0:
        return {"rmse": None, "bias": None, "cells": 0}
    error = np.where(valid, prediction - reference, 0)
    return {
        "rmse": float(np.sqrt((weights * error**2).sum() / total)),
        "bias": float((weights * error).sum() / total),
        "cells": int(valid.sum()),
    }


def limits(*arrays, symmetric=False):
    finite = np.concatenate([np.asarray(a)[np.isfinite(a)].ravel() for a in arrays])
    if not len(finite):
        return (-1, 1) if symmetric else (0, 1)
    low, high = np.percentile(finite, [2, 98])
    if symmetric:
        bound = max(abs(low), abs(high), 1e-8)
        return -bound, bound
    if low == high:
        return low - 0.5, high + 0.5
    return float(low), float(high)


def interval(origin, index):
    start = date.fromisoformat(origin) + timedelta(days=5 * index)
    end = start + timedelta(days=4)
    return f"Day {5 * (index + 1)} · {start.isoformat()}–{end.isoformat()}"


class Catalog:
    def __init__(self, root, extra_catalogs=(), _ancestors=()):
        self.root = Path(root).resolve()
        if self.root in _ancestors:
            raise ValueError("Catalog registry contains a cycle")
        self.meta = json.loads((self.root / "catalog.json").read_text())
        if self.meta["schema_version"] != 1:
            raise ValueError("Unsupported viewer catalog schema")
        self.lat = np.array(self.meta["lat"])
        self.lon = np.array(self.meta["lon"])
        self.depths = np.array(self.meta["depths"])
        self.children = []
        for path in [*self.meta.get("extra_catalogs", []), *extra_catalogs]:
            child_root = self.root / path
            expected = self.meta.get("extra_catalog_hashes", {}).get(path)
            if (
                expected
                and hashlib.sha256(
                    (child_root / "catalog.json").read_bytes()
                ).hexdigest()
                != expected
            ):
                raise ValueError("Child catalog checksum differs")
            self.children.append(
                Catalog(child_root, _ancestors=(*_ancestors, self.root))
            )
        self.all_models = dict(self.meta.get("models", {}))
        for child in self.children:
            if self.all_models.keys() & child.all_models.keys():
                raise ValueError("Model IDs must be unique across catalogs")
            self.all_models.update(child.all_models)

    def for_model(self, model):
        """Route a saved model to its immutable geometry and arrays."""
        if model in self.meta.get("models", {}):
            return self
        for child in self.children:
            if model in child.all_models:
                return child.for_model(model)
        raise KeyError(model)

    def geometry(self, mode):
        if mode == "heat" and "heat_lat" in self.meta:
            return np.asarray(self.meta["heat_lat"]), np.asarray(self.meta["heat_lon"])
        return self.lat, self.lon

    @property
    def depth_interfaces(self):
        interfaces = np.asarray(
            self.meta.get(
                "depth_interfaces",
                [0, 5, 15, 30, 50, 80, 130, 200, 300, 450, 650, 900, 1200, 1600, 2100],
            )
        )
        if not np.array_equal((interfaces[:-1] + interfaces[1:]) / 2, self.depths):
            raise ValueError("Depth interfaces do not match saved layer centers")
        return interfaces

    def models_for(self, mode, origin):
        if mode == "interior" and "interior_examples" in self.meta:
            return set(self.meta["interior_examples"].get(origin, {}).get("models", {}))
        return {
            model
            for model, record in self.meta.get("records", {})
            .get(origin, {})
            .get("models", {})
            .items()
            if mode in record
        }

    def available_modes(self, model):
        modes = set()
        for origin in self.meta.get("origins", []):
            for mode in ("surface", "interior", "forecast_interior", "heat"):
                if model in self.models_for(mode, origin):
                    modes.add(mode)
        return modes

    @lru_cache(maxsize=64)
    def array(self, name):
        path = (self.root / name).resolve()
        if path.parent != self.root or name not in self.meta["files"]:
            raise ValueError("Array is not in the catalog")
        result = np.load(path, mmap_mode="r", allow_pickle=False)
        info = self.meta["files"][name]
        if list(result.shape) != info["shape"] or str(result.dtype) != info["dtype"]:
            raise ValueError(f"Array metadata mismatch: {name}")
        return result

    def values(self, mode, model, origin, reference="observations"):
        owner = self.for_model(model) if self.all_models else self
        if reference in self.all_models and self.for_model(reference) is not owner:
            raise ValueError("Comparison models must use the same catalog geometry")
        if owner is not self:
            return owner.values(mode, model, origin, reference)
        if mode == "interior" and "interior_examples" in self.meta:
            record = self.meta["interior_examples"][origin]
            prediction = self.array(record["models"][model])
            comparison = self.array(
                record["references"][reference]
                if reference in record["references"]
                else record["models"][reference]
            )
            return prediction, comparison
        record = self.meta["records"][origin]
        prediction = self.array(record["models"][model][mode])
        if reference == "persistence":
            if mode == "forecast_interior":
                name = self.meta["interior_examples"][origin]["models"][model]
                return prediction, np.broadcast_to(self.array(name), prediction.shape)
            kind = "heat" if mode == "report_heat" else mode
            comparison = self.array(record["models"][model][kind + "_persistence"])
        elif reference == "observations":
            comparison = self.array(
                record[f"{'heat' if mode == 'report_heat' else mode}_reference"]
            )
        elif reference == "climatology":
            if mode != "interior":
                raise ValueError("December climatology is only an interior reference")
            comparison = self.array(self.meta["climatology"])
        else:
            comparison = self.array(record["models"][reference][mode])
        return prediction, comparison


def raster_rows(values, wet, lat, lon, palette, low, high):
    """Exact native row strips, retaining irregular latitude cell boundaries.

    RGB pixels encode display colors only; the separate values keep every native
    physical value for hover/testing. Land is gray and missing wet cells white.
    """
    values, wet = np.asarray(values), np.asarray(wet, bool)
    if values.shape != (len(lat), len(lon)) or wet.shape != values.shape:
        raise ValueError("Raster values/mask/coordinates differ")
    longitude_edges = edges(lon, (0, 360))
    if not np.allclose(np.diff(longitude_edges), np.diff(longitude_edges)[0]):
        raise ValueError("Row-strip rendering requires uniform longitude cells")
    latitude_edges = edges(lat, (-90, 90))
    colors = np.array(
        [
            [int(color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)]
            for color in palette
        ],
        dtype=np.uint8,
    )
    valid = wet & np.isfinite(values)
    normalized = np.where(valid, (values - low) / max(high - low, 1e-12), 0)
    indices = np.clip(np.floor(normalized * len(colors)), 0, len(colors) - 1).astype(
        int
    )
    rgba = np.empty((*values.shape, 4), dtype=np.uint8)
    rgba[..., :3] = colors[indices]
    rgba[..., 3] = 255
    rgba[~wet] = [213, 219, 224, 255]
    rgba[wet & ~valid] = [255, 255, 255, 255]
    packed = rgba.view(np.uint32).reshape(values.shape)
    return dict(
        image=[row[None] for row in packed],
        values=[row[None] for row in values],
        x=np.zeros(len(lat)),
        y=latitude_edges[:-1],
        dw=np.full(len(lat), 360.0),
        dh=np.diff(latitude_edges),
    )
