# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Plot paired endpoint fields from a continuous rollout, without crop selection."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("source", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
control = np.load(a.source / "control.npz")
gated = np.load(a.source / "gated.npz")
meta = json.loads((a.source / "metrics.json").read_text())
for key in ("initial", "truth_5", "truth_360", "mask", "lat", "lon"):
    np.testing.assert_array_equal(control[key], gated[key])
for store in (control, gated):
    for key in store.files:
        assert np.isfinite(store[key]).all()
labels = [
    "SSH",
    "Surface temperature",
    "Surface salinity",
    "Surface U",
    "Surface V",
    "Temperature at 1050 m",
    "Salinity at 1050 m",
]
units = ["m", "°C", "psu", "m/s", "m/s", "°C", "psu"]
records = []
for slot, (channel, title, unit) in enumerate(zip(meta["channels"], labels, units)):
    for regional in (False, True) if slot in (1, 5, 6) else (False,):
        crop = np.s_[75:105, 200:245] if regional else np.s_[:, :]
        lat = control["lat"][75:105] if regional else control["lat"]
        lon = control["lon"][200:245] if regional else control["lon"]
        wet = control["mask"][slot][crop]
        weights = np.cos(np.deg2rad(lat))[:, None] * wet
        for day in (5, 360):
            truth = control[f"truth_{day}"][slot][crop]
            base = control[f"prediction_{day}"][slot][crop]
            pred = gated[f"prediction_{day}"][slot][crop]
            eb, eg = base - truth, pred - truth
            lo, hi = np.quantile(
                np.concatenate([v[wet] for v in (truth, base, pred)]), [0.01, 0.99]
            )
            lim = np.quantile(np.abs(np.concatenate([v[wet] for v in (eb, eg)])), 0.99)
            # Explicit integer-pixel axes; layout engines/colorbars must not resize them.
            scale = 6 if regional else 1
            height, width = np.array(wet.shape) * scale
            extent = (lon[0] - 0.5, lon[-1] + 0.5, lat[0] - 0.5, lat[-1] + 0.5)
            for kind, vals, names in (
                (
                    "fields",
                    (truth, base, pred),
                    ("Truth", "Ungated control", "Native gate"),
                ),
                ("errors", (eb, eg), ("Control minus truth", "Gate minus truth")),
            ):
                total_width = 65 + len(vals) * (width + 35) + 85
                total_height = height + 125
                fig = plt.figure(
                    figsize=(total_width / 100, total_height / 100), dpi=100
                )
                date = meta["models"]["control"][str(day)]["date"][:10]
                fig.suptitle(
                    f"{title} — day {day} ({date})\nFinal EMA; origin {meta['origin_time'][:10]}",
                    fontsize=11,
                )
                for col, (val, label) in enumerate(zip(vals, names)):
                    ax = fig.add_axes(
                        (
                            (65 + col * (width + 35)) / total_width,
                            45 / total_height,
                            width / total_width,
                            height / total_height,
                        )
                    )
                    im = ax.imshow(
                        np.where(wet, val, np.nan),
                        origin="lower",
                        extent=extent,
                        interpolation="nearest",
                        resample=False,
                        aspect="auto",
                        cmap="RdBu_r" if kind == "errors" else "viridis",
                        vmin=-lim if kind == "errors" else lo,
                        vmax=lim if kind == "errors" else hi,
                    )
                    ax.set_title(label, fontsize=9)
                    ax.set_xlabel("Longitude (°E)", fontsize=8)
                    ax.tick_params(labelsize=7)
                    if col == 0:
                        ax.set_ylabel("Latitude (°N)", fontsize=8)
                    # Keep the outermost map pixels intact.
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    fig.canvas.draw()
                    np.testing.assert_allclose(
                        ax.get_window_extent().bounds,
                        (65 + col * (width + 35), 45, width, height),
                        atol=1e-8,
                    )
                cax = fig.add_axes(
                    (
                        (total_width - 65) / total_width,
                        45 / total_height,
                        12 / total_width,
                        height / total_height,
                    )
                )
                cb = fig.colorbar(im, cax=cax)
                cb.set_label(unit, fontsize=8)
                cb.ax.tick_params(labelsize=7)
                filename = f"{channel}-{'regional' if regional else 'global'}-day{day}-{kind}.png"
                fig.savefig(
                    a.output / filename, dpi=100, pil_kwargs={"compress_level": 9}
                )
                plt.close(fig)
            records.append(
                dict(
                    channel=channel,
                    day=day,
                    region="equatorial-Pacific" if regional else "global",
                    control_rmse=float(
                        np.sqrt((eb**2 * weights).sum() / weights.sum())
                    ),
                    gated_rmse=float(np.sqrt((eg**2 * weights).sum() / weights.sum())),
                )
            )
(a.output / "map-metrics.json").write_text(
    json.dumps(dict(origin_time=meta["origin_time"], rows=records), indent=2) + "\n"
)
print(json.dumps(records, indent=2))
