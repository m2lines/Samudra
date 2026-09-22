#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Render native-cell four-panel maps with common scales across runs and dates."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(path, output, dates, scale, limit):
    output.mkdir(parents=True, exist_ok=True)
    with np.load(path) as data:
        keys = {
            str(data[k]).split()[0]: k.removesuffix("_date")
            for k in data.files
            if k.endswith("_date")
        }
        latitude = data["latitude"]
        mask = data["mask"]
        std = float(data["std"])
        weights = mask * np.cos(np.deg2rad(latitude))[:, None]
        weights = weights / weights.sum()
        for day in dates:
            idx = keys[day]
            t, c, p = [
                np.where(mask, data[idx + "_" + field], np.nan) * std
                for field in ["truth", "climatology", "prediction"]
            ]
            a, b, e = t - c, p - c, p - t

            def avg(v):
                return float(np.nansum(v * weights))

            def rms(v):
                return avg(v * v) ** 0.5

            corr = avg((a - avg(a)) * (b - avg(b))) / rms(a - avg(a)) / rms(b - avg(b))
            h, w = mask.shape
            pw, ph = w * scale, h * scale
            left, gap, right, top, between, bottom = 65, 65, 25, 100, 85, 150
            width = left + 2 * pw + gap + right
            height = top + 2 * ph + between + bottom
            fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
            fig.text(
                0.5,
                1 - 24 / height,
                f"550 m salinity | {day} | {path.parent.name}",
                ha="center",
                fontsize=13,
            )
            fig.text(
                0.5,
                1 - 48 / height,
                f"RMSE {rms(e):.4f} | Climatology RMSE {rms(a):.4f} | Anomaly correlation {corr:.3f}",
                ha="center",
                fontsize=10,
            )
            for k, (title, field) in enumerate(
                [
                    ("True anomaly", a),
                    ("Predicted anomaly", b),
                    ("Error: prediction − truth", e),
                    ("Climatology error", -a),
                ]
            ):
                row, col = divmod(k, 2)
                x = left + col * (pw + gap)
                y = height - top - ph - row * (ph + between)
                ax = fig.add_axes((x / width, y / height, pw / width, ph / height))
                im = ax.imshow(
                    np.ma.masked_invalid(field),
                    origin="lower",
                    interpolation="nearest",
                    resample=False,
                    cmap="RdBu_r",
                    vmin=-limit,
                    vmax=limit,
                    aspect="equal",
                )
                ax.set_facecolor("#ddd")
                ax.set_title(title, fontsize=11, pad=9)
                ax.set_xticks(
                    [-0.5, w * 0.25 - 0.5, w * 0.5 - 0.5, w * 0.75 - 0.5, w - 0.5],
                    ["0", "90", "180", "270", "360"],
                )
                ax.set_yticks(
                    [
                        int(np.argmin(np.abs(latitude - v)))
                        for v in [-60, -30, 0, 30, 60]
                    ],
                    ["−60", "−30", "0", "30", "60"],
                )
                ax.tick_params(labelsize=9)
                ax.set_xlabel("Longitude", fontsize=9)
                ax.set_ylabel("Latitude", fontsize=9)
            cax = fig.add_axes(
                (left / width, 76 / height, (2 * pw + gap) / width, 15 / height)
            )
            fig.colorbar(
                im,
                cax=cax,
                orientation="horizontal",
                label="Salinity difference; same limits across all panels, dates, and runs",
            )
            fig.text(
                0.5,
                8 / height,
                f"{w} × {h} grid → {pw} × {ph} pixels per panel; no interpolation",
                ha="center",
                fontsize=9,
            )
            fig.canvas.draw()
            for ax in fig.axes[:4]:
                box = ax.get_window_extent()
                assert abs(box.width - pw) < 1e-6 and abs(box.height - ph) < 1e-6
            fig.savefig(output / f"{path.parent.name}-{day}-{scale}x.png", dpi=100)
            plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--maps", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--dates",
        nargs="+",
        default=[
            "2014-11-04",
            "2015-02-02",
            "2015-05-03",
            "2015-08-01",
            "2018-11-14",
            "2022-11-24",
        ],
    )
    p.add_argument("--scale", type=int, choices=[1, 2, 3], default=2)
    # Physical equivalent of the original fixed 2018 true-anomaly 98th percentile.
    p.add_argument(
        "--limit", type=float, default=0.2043664735555648 * 0.709178633744918
    )
    args = p.parse_args()
    plot(args.maps, args.output, args.dates, args.scale, args.limit)


if __name__ == "__main__":
    main()
