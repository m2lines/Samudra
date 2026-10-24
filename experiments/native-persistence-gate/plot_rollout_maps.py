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
        fig, axes = plt.subplots(2, 5, figsize=(16, 6), layout="constrained")
        for row, day in enumerate((5, 360)):
            truth = control[f"truth_{day}"][slot][crop]
            base = control[f"prediction_{day}"][slot][crop]
            pred = gated[f"prediction_{day}"][slot][crop]
            eb, eg = base - truth, pred - truth
            lo, hi = np.quantile(
                np.concatenate([v[wet] for v in (truth, base, pred)]), [0.01, 0.99]
            )
            lim = np.quantile(np.abs(np.concatenate([v[wet] for v in (eb, eg)])), 0.99)
            vals = (truth, base, pred, eb, eg)
            for col, (ax, val, label) in enumerate(
                zip(
                    axes[row],
                    vals,
                    (
                        "Truth",
                        "Ungated control",
                        "Native gate",
                        "Control error",
                        "Gated error",
                    ),
                )
            ):
                error = col >= 3
                im = ax.pcolormesh(
                    lon,
                    lat,
                    np.where(wet, val, np.nan),
                    shading="nearest",
                    cmap="RdBu_r" if error else "viridis",
                    vmin=-lim if error else lo,
                    vmax=lim if error else hi,
                    rasterized=True,
                )
                ax.set_title(label, fontsize=10)
                ax.set_xlabel("Longitude (°E)", fontsize=8)
                if col == 0:
                    date = meta["models"]["control"][str(day)]["date"][:10]
                    ax.set_ylabel(f"Day {day}: {date}\nLatitude (°N)", fontsize=8)
                ax.tick_params(labelsize=7)
                if col in (2, 4):
                    cb = fig.colorbar(
                        im,
                        ax=axes[row, :3] if col == 2 else axes[row, 3:],
                        shrink=0.7,
                        pad=0.02,
                    )
                    cb.set_label(unit, fontsize=8)
                    cb.ax.tick_params(labelsize=7)
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
        fig.suptitle(
            f"{title} — final EMA checkpoints, one continuous rollout\nOrigin {meta['origin_time'][:10]}; shared model/error scales within each row",
            fontsize=12,
        )
        filename = f"{channel}-regional.png" if regional else f"{channel}-global.jpg"
        fig.savefig(
            a.output / filename,
            dpi=85,
            pil_kwargs={"compress_level": 9}
            if regional
            else {"quality": 85, "optimize": True},
        )
        plt.close(fig)
(a.output / "map-metrics.json").write_text(
    json.dumps(dict(origin_time=meta["origin_time"], rows=records), indent=2) + "\n"
)
print(json.dumps(records, indent=2))
