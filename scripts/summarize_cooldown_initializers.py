#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Reduce frozen annual initial states, without claiming monthly target accuracy."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

DEPTHS = [
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
]


def mean(value, weight):
    valid = np.isfinite(value) & (weight > 0)
    return float(
        np.sum(np.where(valid, value * weight, 0)) / np.sum(np.where(valid, weight, 0))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    plan = json.loads((root / "plan.json").read_text())
    with np.load(Path(plan["data"]) / "grid.npz") as grid:
        mask, lat = grid["mask"], grid["lat"]
    area = np.cos(np.deg2rad(lat))[:, None]
    rows, heat, maps = [], [], {}
    for family in ("U-global", "U-multitask-early"):
        for weights in ("raw", "ema"):
            with np.load(
                root / "results" / family / "S0" / weights / "2013-11-01.npz"
            ) as data:
                base = data["initial"][-1]
                base_ohc = data["persistence_ohc"]
            for branch in ("S0", "A", "B", "C", "D"):
                path = root / "results" / family / branch / weights / "2013-11-01.npz"
                with np.load(path) as data:
                    state, ohc = data["initial"][-1], data["persistence_ohc"]
                    for variable, offset in [
                        ("temperature", 38),
                        ("salinity", 57),
                        ("u", 0),
                        ("v", 19),
                    ]:
                        for i, depth in enumerate(DEPTHS):
                            w = area * mask[offset + i]
                            value, reference = state[offset + i], base[offset + i]
                            rows.append(
                                dict(
                                    family=family,
                                    weights=weights,
                                    branch=branch,
                                    variable=variable,
                                    depth_m=depth,
                                    mean=mean(value, w),
                                    mean_change=mean(value - reference, w),
                                    rms_change=mean((value - reference) ** 2, w) ** 0.5,
                                )
                            )
                    for k, layer in enumerate(["0_700", "700_2000"]):
                        w = area * np.isfinite(ohc[k]) * np.isfinite(base_ohc[k])
                        heat.append(
                            dict(
                                family=family,
                                weights=weights,
                                branch=branch,
                                layer=layer,
                                mean_change_J_m2=mean(ohc[k] - base_ohc[k], w),
                                rms_change_J_m2=mean((ohc[k] - base_ohc[k]) ** 2, w)
                                ** 0.5,
                            )
                        )
                    if weights == "ema":
                        maps[f"{family}_{branch}_temperature105m"] = np.where(
                            mask[43], state[43] - base[43], np.nan
                        )
    for name, records in [
        ("initializer-profiles.csv", rows),
        ("initializer-ohc.csv", heat),
    ]:
        with (root / name).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    np.savez_compressed(root / "initializer-maps.npz", **maps)


if __name__ == "__main__":
    main()
