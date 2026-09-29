#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Render lossless native-grid comparison and initializer-evolution panels."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from summarize_observation_missingness import NAMES  # type: ignore[import-not-found]

FIELDS = [
    (0, "SST", "°C", (-2, 32), "turbo"),
    (1, "Temperature at 550 m", "°C", (-2, 18), "turbo"),
    (2, "Surface salinity", "psu", (28, 38), "viridis"),
    (4, "Surface zonal velocity", "m/s", (-0.5, 0.5), "RdBu_r"),
    (5, "Surface meridional velocity", "m/s", (-0.5, 0.5), "RdBu_r"),
    (6, "SSH", "m", (-1.8, 1), "RdBu_r"),
]


def panel_plot(panels, title, field, path):
    channel, _, unit, limits, cmap = field
    cols = min(3, len(panels))
    rows = math.ceil(len(panels) / cols)
    width, height = 40 + 410 * cols, 110 + 225 * rows
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    tiles = []
    stats = []
    for i, (label, value, grid) in enumerate(panels):
        order = np.argsort((grid["lon"] + 180) % 360 - 180)
        z = np.where(grid["mask"][channel], value[channel], np.nan)[:, order]
        left, top = 45 + 410 * (i % cols), 55 + 225 * (i // cols)
        ax = fig.add_axes(
            (left / width, (height - top - 180) / height, 360 / width, 180 / height)
        )
        im = ax.imshow(
            z,
            origin="lower",
            cmap=cmap,
            vmin=limits[0],
            vmax=limits[1],
            interpolation="none",
        )
        ax.set_title(label, fontsize=8)
        ax.set_xticks([0, 180, 359], ["180°W", "0°", "180°E"])
        ax.set_yticks([0, 90, 179], ["90°S", "0°", "90°N"])
        ax.tick_params(length=0, labelsize=6)
        np.testing.assert_allclose(ax.get_window_extent().size, [360, 180], atol=1e-6)
        rgba = im.to_rgba(z, bytes=True)
        tiles.append((left, top, np.where(rgba[..., 3:] > 0, rgba[..., :3], 209)[::-1]))
        wet = z[np.isfinite(z)]
        stats.append(
            {
                "label": label,
                "min": float(wet.min()),
                "max": float(wet.max()),
                "below_color_limit": int((wet < limits[0]).sum()),
                "above_color_limit": int((wet > limits[1]).sum()),
            }
        )
    fig.suptitle(title, fontsize=11)
    cax = fig.add_axes((0.25, 45 / height, 0.5, 12 / height))
    fig.colorbar(im, cax=cax, orientation="horizontal", label=unit)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    raster = Image.open(path).convert("RGB")
    for left, top, tile in tiles:
        raster.paste(Image.fromarray(tile), (left, top))
    raster.save(path, optimize=True)
    pixels = np.array(Image.open(path))
    for left, top, tile in tiles:
        np.testing.assert_array_equal(pixels[top : top + 180, left : left + 360], tile)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    arms = [arm for arm in NAMES if (args.arrays / arm / "manifest.json").exists()]
    audit: dict[str, Any] = {"inputs": {}, "figures": {}, "natural_missing": {}}
    for arm in arms:
        root = args.arrays / arm
        manifest = json.loads((root / "manifest.json").read_text())
        for name, entry in manifest["files"].items():
            sha = hashlib.sha256((root / name).read_bytes()).hexdigest()
            assert sha == entry["sha256"]
            audit["inputs"][arm + "/" + name] = sha
    grids = {arm: dict(np.load(args.arrays / arm / "grid.npz")) for arm in arms}
    for arm in arms:
        for key in ("lat", "lon", "mask", "channels"):
            np.testing.assert_array_equal(grids[arm][key], grids[arms[0]][key])
    for arm in arms:
        grid = grids[arm]
        audit["natural_missing"][arm] = {}
        for path in sorted((args.arrays / arm).glob("best-*-natural.npz")):
            data = np.load(path)
            rows = {}
            for channel, variable, visible_channel in ((0, "sst", 0), (6, "ssh", 1)):
                missing = grid["mask"][channel] & ~data["visible"][
                    visible_channel
                ].astype(bool)
                for region, latitude_mask in (
                    ("global", np.ones(180, dtype=bool)),
                    ("north_of_60", grid["lat"] > 60),
                    ("south_of_60", grid["lat"] < -60),
                ):
                    selected = missing & latitude_mask[:, None]
                    value = data["initial"][channel][selected]
                    weight = np.broadcast_to(
                        np.cos(np.deg2rad(grid["lat"]))[:, None], selected.shape
                    )[selected]
                    rows[variable + "/" + region] = {
                        "count": int(selected.sum()),
                        "area_weighted_mean": float(np.average(value, weights=weight))
                        if len(value)
                        else None,
                        "percentiles_0_1_50_99_100": np.percentile(
                            value, [0, 1, 50, 99, 100]
                        ).tolist()
                        if len(value)
                        else None,
                    }
            audit["natural_missing"][arm][path.stem] = rows
    coverage_arm = "masked-scratch" if "masked-scratch" in arms else arms[0]
    for month in ("2013-11", "2014-07"):
        data = np.load(args.arrays / coverage_arm / ("best-" + month + "-natural.npz"))
        panels = []
        for variable, visible_channel, grid_channel in (("SST", 0, 0), ("SSH", 1, 6)):
            grid = dict(grids[coverage_arm])
            grid["mask"] = grid["mask"].copy()
            grid["mask"][0] = grid["mask"][grid_channel]
            value = np.zeros_like(data["initial"])
            value[0] = data["visible"][visible_channel]
            panels.append((variable, value, grid))
        name = "surface-coverage-" + month + ".png"
        audit["figures"][name] = panel_plot(
            panels,
            "Surface inputs at initialization: " + month,
            (0, "Coverage", "0 = missing; 1 = observed; gray = land", (0, 1), "Greens"),
            args.output / name,
        )
    for field in FIELDS:
        channel, variable = field[:2]
        for state in ("initial", "day30"):
            panels = [
                (NAMES[a], np.load(args.arrays / a / "best.npz")[state][0], grids[a])
                for a in arms
            ]
            name = f"selected-{state}-channel{channel}.png"
            audit["figures"][name] = panel_plot(
                panels,
                f"{variable}: selected checkpoints, November 2013 / {state}",
                field,
                args.output / name,
            )
        for origin in ("2015-01-01", "2018-01-01", "2021-01-01"):
            panels = [
                (
                    NAMES[a],
                    np.load(args.arrays / a / ("annual-" + origin + ".npz"))["state"][
                        1
                    ],
                    grids[a],
                )
                for a in arms
            ]
            name = f"day365-{origin}-channel{channel}.png"
            audit["figures"][name] = panel_plot(
                panels,
                f"{variable}: day 365 from {origin} (diagnostic)",
                field,
                args.output / name,
            )
        for arm in arms:
            checkpoints = [
                "joint-00000",
                "joint-00010",
                "joint-00100",
                "joint-01000",
                "joint-04000",
                "before-observation-finish",
                "joint-06500",
                "joint-08000",
                "joint-16000",
                "best",
            ]
            checkpoints = [
                c for c in checkpoints if (args.arrays / arm / (c + ".npz")).exists()
            ]
            panels = [
                (
                    "Random initialization"
                    if c == "joint-00000"
                    else "Selected"
                    if c == "best"
                    else "Before obs-only finish"
                    if c == "before-observation-finish"
                    else str(int(c.split("-")[1])) + " observation updates",
                    np.load(args.arrays / arm / (c + ".npz"))["initial"][0],
                    grids[arm],
                )
                for c in checkpoints
            ]
            name = f"evolution-{arm}-channel{channel}.png"
            audit["figures"][name] = panel_plot(
                panels,
                f"{NAMES[arm]}: initialized {variable}, November 2013",
                field,
                args.output / name,
            )
    (args.output / "map-audit.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
