# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Whole-validation quantization floors versus persistence, without a model run."""

import json
import math
from pathlib import Path

import numpy as np

from .data import DEPTHS, TEMPERATURES
from .model import Vocabulary


def audit_precision(source, config, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    support = json.loads(Path(config["support"]).read_text())
    for key in ("metadata_sha256", "mean", "std"):
        if support["source"][key] != source.receipt()[key]:
            raise ValueError(f"Support/data mismatch: {key}")
    if not support["complete_training_scan"]:
        raise ValueError("Precision audit requires full training-derived support")
    absolute = Vocabulary(support["absolute_bounds"], config["model"]["bins"])
    residual = Vocabulary(support["residual_bounds"], config["model"]["bins"])
    centers = [v.centers.numpy() for v in (absolute, residual)]
    edges = [v.edges.numpy() for v in (absolute, residual)]
    accumulated = np.zeros((19, 5), dtype=np.float64)
    origins = source.origins["validation"]
    for count, origin in enumerate(origins, 1):

        def measure(depth):
            wet = source.wet[depth]
            weights = source.area[wet]
            mean, std = source.mean[38 + depth], source.std[38 + depth]
            current = (
                np.asarray(source.zarr[TEMPERATURES[depth]][int(origin)])[wet] - mean
            ) / std
            target = (
                np.asarray(source.zarr[TEMPERATURES[depth]][int(origin) + 1])[wet]
                - mean
            ) / std
            if not np.isfinite(current).all() or not np.isfinite(target).all():
                raise ValueError(
                    f"Nonfinite validation temperatures at {origin}/{depth}"
                )
            delta = target - current
            predictions = []
            outside = []
            for values, c, e in zip((target, delta), centers, edges, strict=True):
                index = np.searchsorted(e[depth, 1:-1], values)
                predictions.append(c[depth, index])
                outside.append(
                    np.average(
                        (values < e[depth, 0]) | (values > e[depth, -1]),
                        weights=weights,
                    )
                )
            errors = [
                predictions[0].astype(np.float64) - target,
                predictions[1].astype(np.float64) - delta,
                current.astype(np.float64) - target,
            ]
            return [
                float(np.average(error**2, weights=weights) * float(std) ** 2)
                for error in errors
            ] + outside

        rows = list(source.pool.map(measure, range(19)))
        accumulated += np.asarray(rows)
        print(json.dumps({"precision_dates": count, "total": len(origins)}), flush=True)
    result = {
        "source": source.receipt(),
        "support": support,
        "scope": "all wet cells, all valid validation windows; cell-area weights, equal dates",
        "validation_dates": len(origins),
        "per_depth": [],
    }
    for depth, row in enumerate(accumulated / len(origins)):
        result["per_depth"].append(
            {
                "depth": depth,
                "depth_m": float(DEPTHS[depth]),
                "absolute_oracle_rmse_c": math.sqrt(row[0]),
                "residual_oracle_rmse_c": math.sqrt(row[1]),
                "persistence_rmse_c": math.sqrt(row[2]),
                "absolute_oracle_to_persistence": math.sqrt(row[0] / row[2])
                if row[2]
                else None,
                "absolute_outside_area_fraction": row[3],
                "residual_outside_area_fraction": row[4],
            }
        )
    (destination / "precision.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
