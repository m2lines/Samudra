#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Render frozen gap interventions with one pixel per grid cell."""

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

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--bundle-root", type=Path, required=True)
parser.add_argument("--state-map-grid", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root, out = args.bundle_root, args.output
out.mkdir(parents=True, exist_ok=True)
bundle_manifest = json.loads((root / "gap-report-bundle.json").read_text())
for name, record in bundle_manifest["files"].items():
    path = root / name
    if (
        path.stat().st_size != record["bytes"]
        or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]
    ):
        raise ValueError("Changed gap report input: " + name)
g = np.load(args.state_map_grid)
lat, lon = g["lat"], g["lon"]
order = np.argsort((lon + 180) % 360 - 180)
audit: dict[str, Any] = {"maps": []}


def maps(name, title, panels, cmap, limits, unit, cols=3, scale=1, regional=False):
    rows = (len(panels) + cols - 1) // cols
    cellw = 400 if not regional else 230
    rowh = 225 if not regional else 110
    w = 50 + cols * cellw
    h = 95 + rows * rowh + 70
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    tiles = []
    for i, (label, z) in enumerate(panels):
        left = 45 + (i % cols) * cellw
        top = 85 + (i // cols) * rowh
        if z is None:
            continue
        z = np.asarray(z)
        if not regional:
            z = z[:, order]
        ph, pw = np.array(z.shape) * scale
        ax = fig.add_axes((left / w, (h - top - ph) / h, pw / w, ph / h))
        im = ax.imshow(
            z,
            origin="lower",
            cmap=cmap,
            vmin=limits[0],
            vmax=limits[1],
            interpolation="none",
            aspect="equal",
        )
        ax.set_title(label, fontsize=8)
        if regional:
            ax.set_axis_off()
        else:
            ax.set_xticks([0, 180, 359], ["−180°", "0°", "180°"])
            ax.set_yticks([0, 90, 179], ["90°S", "0°", "90°N"])
            ax.tick_params(length=0, labelsize=6)
            for spine in ax.spines.values():
                spine.set_visible(False)
        np.testing.assert_allclose(ax.get_window_extent().size, [pw, ph], atol=1e-6)
        rgba = im.to_rgba(z, bytes=True)
        tile = np.repeat(
            np.repeat(np.where(rgba[..., 3:] > 0, rgba[..., :3], 209)[::-1], scale, 0),
            scale,
            1,
        )
        tiles.append((left, top, tile))
    cax = fig.add_axes((0.2, 45 / h, 0.6, 12 / h))
    fig.colorbar(im, cax=cax, orientation="horizontal", label=unit, extend="both")
    fig.suptitle(title, fontsize=11, y=1 - 15 / h)
    fig.savefig(out / name, dpi=100)
    plt.close(fig)
    raster = Image.open(out / name).convert("RGB")
    for left, top, tile in tiles:
        raster.paste(Image.fromarray(tile), (left, top))
    raster.save(out / name, optimize=True)
    pixels = np.array(raster)
    for left, top, tile in tiles:
        np.testing.assert_array_equal(
            pixels[top : top + tile.shape[0], left : left + tile.shape[1]], tile
        )
    # Indexed color changes neither geometry nor the constant grid-cell blocks.
    # Use a single image palette to keep report files below the repository limit.
    if (out / name).stat().st_size > 245000:
        raster = raster.quantize(colors=256, dither=Image.Dither.NONE)
        raster.save(out / name, optimize=True)
        pixels = np.array(raster.convert("RGB"))
        for left, top, tile in tiles:
            block = pixels[top : top + tile.shape[0], left : left + tile.shape[1]]
            if scale == 2:
                np.testing.assert_array_equal(
                    block, np.repeat(np.repeat(block[::2, ::2], 2, 0), 2, 1)
                )
    if (out / name).stat().st_size >= 250000:
        (out / name).unlink()
        chunk = cols if rows > 1 else max(1, cols // 2)
        for start in range(0, len(panels), chunk):
            maps(
                Path(name).stem + f"-part{start // chunk + 1}.png",
                title,
                panels[start : start + chunk],
                cmap,
                limits,
                unit,
                cols=min(cols, chunk),
                scale=scale,
                regional=regional,
            )
        return
    audit["maps"].append(
        {
            "file": name,
            "pixels_per_grid_cell": scale,
            "panels": len(tiles),
            "cell_geometry_verified": True,
        }
    )


arms = ["scratch", "sequential", "mixed"]
modes = ["baseline", "initial-zero", "history-zero", "initial-om4", "history-om4"]
short = ["Original", "Initial OM4 fill", "History OM4 fill"]
arrays = {
    (a, m): np.load(root / ("gaps-" + a) / ("2015-01-01-" + m + ".npz"))
    for a in arms
    for m in ["baseline", "initial-om4", "history-om4"]
}
for array in arrays.values():
    np.testing.assert_array_equal(array["lat"], lat)
    np.testing.assert_array_equal(array["lon"], lon)

for ch, maskch, label, cmap, lim, unit in [
    (0, 0, "sst", "turbo", (-2, 32), "degrees C"),
    (1, 6, "ssh", "RdBu_r", (-1.5, 1.5), "m"),
]:
    wet = g["mask"][maskch].astype(bool)
    panels = [
        (text, np.where(wet, arrays["scratch", m]["input_surface"][-1, ch], np.nan))
        for text, m in zip(short[:2], ["baseline", "initial-om4"])
    ]
    maps(
        "gap-initial-" + label + ".png",
        "Initial "
        + label.upper()
        + " · 2015-01-01 · identical surface inputs in all three old models",
        panels,
        cmap,
        lim,
        unit,
        cols=2,
    )
    for day in [30, 365]:
        panels = []
        for a in arms:
            for text, m in zip(short, ["baseline", "initial-om4", "history-om4"]):
                z = arrays[a, m]
                ix = list(z["leads_days"]).index(day)
                panels.append(
                    (a + " · " + text, np.where(wet, z["surface"][ix, ch], np.nan))
                )
        maps(
            f"gap-day{day}-{label}.png",
            f"{label.upper()} · day {day} · rollout from 2015-01-01",
            panels,
            cmap,
            lim,
            unit,
            cols=3,
        )
results: dict[str, Any] = {}
summary: dict[str, Any] = {}
for a in arms:
    results[a] = {}
    summary[a] = {}
    for m in modes:
        ds = [
            json.loads(p.read_text())
            for p in sorted((root / ("gaps-" + a)).glob("*-" + m + ".json"))
        ]
        results[a][m] = [
            {
                k: v
                for k, v in d.items()
                if k not in ["latitude_rmse", "latitude_support", "spectra"]
            }
            for d in ds
        ]
        summary[a][m] = {
            day: {
                key: float(np.mean([d["leads"][day][key] for d in ds]))
                for key in ["sst_rmse", "adt_rmse", "velocity_rmse"]
            }
            for day in ["5", "15", "30", "90", "180", "365"]
        }
fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
for col, a in enumerate(arms):
    for row, key in enumerate(["sst_rmse", "adt_rmse"]):
        for m, text in zip(["baseline", "initial-om4", "history-om4"], short):
            days = [5, 15, 30, 90, 180, 365]
            axes[row, col].plot(
                days,
                [summary[a][m][str(day)][key] for day in days],
                label=text,
                marker=".",
                linewidth=1,
            )
        axes[row, col].set_title(a if row == 0 else "")
        axes[row, col].grid(alpha=0.2)
        axes[row, col].set_ylabel(
            "SST RMSE (degrees C)" if row == 0 else "ADT RMSE (m)"
        )
        if row == 1:
            axes[row, col].set_xlabel("Lead (days)")
axes[0, 0].legend(fontsize=8)
fig.suptitle(
    "Frozen-checkpoint missing-input interventions · mean of three annual cases · 60S–60N"
)
fig.tight_layout()
fig.savefig(out / "gap-lead-errors.png", dpi=100)
plt.close(fig)
audit.update(
    plot_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    grid_bundle_sha256=hashlib.sha256(args.state_map_grid.read_bytes()).hexdigest(),
    source_bundle_sha256=hashlib.sha256(
        (root / "gap-report-bundle.tar").read_bytes()
    ).hexdigest(),
    results=results,
    mean_lead_metrics=summary,
    origins=["2015-01-01", "2018-01-01", "2021-01-01"],
)
(out / "gap-summary.json.gz").write_bytes(
    gzip.compress((json.dumps(audit, indent=2) + "\n").encode(), mtime=0)
)
for p in out.glob("*"):
    if p.name.endswith(".license"):
        continue
    Path(str(p) + ".license").write_text(
        "SPDX-FileCopyrightText: 2026 Samudra Authors\nSPDX-License-Identifier: CC-BY-4.0\n"
    )
print([(p.name, p.stat().st_size) for p in out.iterdir() if p.suffix != ".license"])
