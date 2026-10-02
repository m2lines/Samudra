# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Coordinate-aware access and diagnostics for immutable viewer exports."""

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
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.meta = json.loads((self.root / "catalog.json").read_text())
        if self.meta["schema_version"] != 1:
            raise ValueError("Unsupported viewer catalog schema")
        self.lat = np.array(self.meta["lat"])
        self.lon = np.array(self.meta["lon"])
        self.depths = np.array(self.meta["depths"])

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
        record = self.meta["records"][origin]
        prediction = self.array(record["models"][model][mode])
        if reference == "observations":
            comparison = self.array(record[f"{mode}_reference"])
        elif reference == "climatology":
            if mode != "interior":
                raise ValueError("December climatology is only an interior reference")
            comparison = self.array(self.meta["climatology"])
        else:
            comparison = self.array(record["models"][reference][mode])
        return prediction, comparison
