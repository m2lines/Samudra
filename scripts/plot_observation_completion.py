#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Summarize validation-only completion audits and render native-grid maps."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", default="early-")
    parser.add_argument(
        "--panel", nargs=3, action="append", metavar=("ARM", "STEP", "LABEL")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"inputs": {}, "scores": []}
    for directory in sorted(args.root.glob(args.prefix + "*-completion")):
        signature = json.loads((directory / "COMPLETE.json").read_text())
        assert signature == json.loads((directory / "input.json").read_text())
        for path in sorted(directory.glob("joint-*.json")):
            data = json.loads(path.read_text())
            assert (
                data["checkpoint_sha256"] == signature["checkpoints"][path.stem + ".pt"]
            )
            summary["inputs"][str(path.relative_to(args.root))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for pattern in ("blocks", "polar_caps"):
                for method in ("model", "climatology", "normalized_zero"):
                    row = {
                        "arm": directory.name,
                        "checkpoint": path.stem,
                        "task_counts": data["task_counts"],
                        "pattern": pattern,
                        "method": method,
                    }
                    for key in ("sst/global", "ssh/global"):
                        records = [
                            x[method][key]
                            for x in data["records"]
                            if x["pattern"] == pattern
                        ]
                        weight = sum(x["area_weight"] for x in records)
                        row[key] = (
                            float(
                                np.sqrt(
                                    sum(
                                        x["area_weight"] * x["rmse"] ** 2
                                        for x in records
                                        if x["rmse"] is not None
                                    )
                                    / weight
                                )
                            )
                            if weight
                            else None
                        )
                    summary["scores"].append(row)
    grid = np.load(args.grid)
    panels = args.panel or [
        ("legacy-scratch", "00100", "Legacy scratch: 0 OM4 + 100 obs"),
        ("masked-scratch", "00100", "Masked scratch: 0 OM4 + 100 obs"),
        ("conditioned-mixed-finish", "00050", "Conditioned mixed: 378 OM4 + 50 obs"),
    ]
    width = 30 + 410 * len(panels)
    for pattern in ("natural", "polar_caps"):
        for channel, variable, limits, cmap in (
            (0, "sst", (-2, 32), "turbo"),
            (6, "ssh", (-1.8, 1.0), "RdBu_r"),
        ):
            fig = plt.figure(figsize=(width / 100, 3.2), dpi=100)
            tiles = []
            for index, (arm, step, label) in enumerate(panels):
                path = (
                    args.root
                    / f"{args.prefix}{arm}-completion"
                    / f"joint-{step}-2013-11-{pattern}.npz"
                )
                summary["inputs"][str(path.relative_to(args.root))] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                data = np.load(path)
                np.testing.assert_array_equal(data["lat"], grid["lat"])
                np.testing.assert_array_equal(data["lon"], grid["lon"])
                order = np.argsort((data["lon"] + 180) % 360 - 180)
                z = np.where(grid["mask"][channel], data["initial"][channel], np.nan)[
                    :, order
                ]
                left, top = 45 + 410 * index, 65
                ax = fig.add_axes(
                    (left / width, (320 - top - 180) / 320, 360 / width, 180 / 320)
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
                np.testing.assert_allclose(
                    ax.get_window_extent().size, [360, 180], atol=1e-6
                )
                rgba = im.to_rgba(z, bytes=True)
                tiles.append(
                    (left, top, np.where(rgba[..., 3:] > 0, rgba[..., :3], 209)[::-1])
                )
            label = (
                "natural observation coverage"
                if pattern == "natural"
                else "all observations above |60°| hidden"
            )
            title = f"Initialized {variable.upper()}, November 2013 — {label}"
            if width < 1000:
                title = title.replace(" — ", "\n")
            fig.suptitle(title, fontsize=11)
            cax = fig.add_axes((0.25, 0.09, 0.5, 0.035))
            fig.colorbar(
                im,
                cax=cax,
                orientation="horizontal",
                label="°C" if variable == "sst" else "m",
            )
            target = args.output / f"{args.prefix}{pattern}-{variable}.png"
            fig.savefig(target, dpi=100)
            plt.close(fig)
            raster = Image.open(target).convert("RGB")
            for left, top, tile in tiles:
                raster.paste(Image.fromarray(tile), (left, top))
            raster.save(target, optimize=True)
            pixels = np.array(Image.open(target))
            for left, top, tile in tiles:
                np.testing.assert_array_equal(
                    pixels[top : top + 180, left : left + 360], tile
                )
    (args.output / (args.prefix + "completion-summary.json.gz")).write_bytes(
        gzip.compress((json.dumps(summary, indent=2) + "\n").encode(), mtime=0)
    )


if __name__ == "__main__":
    main()
