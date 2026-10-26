#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Compare saved shared-ERA5 and OBS-only validation records; no inference."""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "docs/experiments/observation-d"
saved = json.loads((ROOT / "assets/shared-era5-2026-10-26/metrics.json").read_text())
old = json.loads((ROOT / "assets/monthly-replay-2026-10-20/results.json").read_text())
OUT = ROOT / "assets/shared-era5-vs-obs-2026-10-26"
OUT.mkdir(exist_ok=True)
norm = old["normalization"]
keys = list(norm)


def shared(name):
    rows = []
    for filename, payload in saved[name].items():
        match = re.fullmatch(r"validation-(\d+)\.json", filename)
        if not match:
            continue
        assert payload["origins"] == old["origins"]
        assert payload["members"] == 8 and payload["sampler_steps"] == 32
        assert (
            payload["readout"]
            == "five-day surface mean plus direct-monthly interior; top OHC layer from mean surface"
        )
        m = payload["metrics"]
        rows.append(
            dict(
                step=int(match[1]),
                metrics=m,
                blend=float(np.mean([m[k] / norm[k] for k in keys])),
            )
        )
    return sorted(rows, key=lambda x: x["step"])


s8 = shared("shared8k")
s4 = [r for r in s8 if r["step"] <= 4000] + shared("shared4k")
e4 = [r for r in old["curves"]["E"] if r["step"] <= 2000] + old["curves"]["E-extend"]
curves = {
    "Shared ERA5 · 4k + 1k": s4,
    "Shared ERA5 · 8k + 1k": s8,
    "OBS-only E-extend · 4k + 1k": e4,
}
colors = ["#0072B2", "#009E73", "#D55E00"]
boundaries = [4000, 8000, 4000]
for rows in curves.values():
    assert len({r["step"] for r in rows}) == len(rows)
    for r in rows:
        assert (
            abs(r["blend"] - np.mean([r["metrics"][k] / norm[k] for k in keys])) < 1e-10
        )


def xval(step, exposure, obs_only, boundary):
    if not exposure or obs_only:
        return step
    return min(step, boundary) / 2 + max(step - boundary, 0)


def draw(ax, key, exposure=False, late=False):
    for (name, rows), color, boundary in zip(
        curves.items(), colors, boundaries, strict=True
    ):
        obs_only = name.startswith("OBS-only")
        rows = [r for r in rows if not late or r["step"] >= 1500]

        def value(r):
            v = r["blend"] if key == "blend" else r["metrics"][key]
            return v / 1e9 if key.startswith("ohc") else v

        for cooldown, style in [(False, "-"), (True, "--")]:
            subset = [
                r
                for r in rows
                if (r["step"] >= boundary if cooldown else r["step"] <= boundary)
            ]
            ax.plot(
                [xval(r["step"], exposure, obs_only, boundary) for r in subset],
                [value(r) for r in subset],
                style,
                color=color,
                lw=1.6,
                marker=".",
                ms=3,
                label=name if not cooldown else None,
            )
        br = next((r for r in rows if r["step"] == boundary), None)
        if br:
            ax.plot(
                xval(boundary, exposure, obs_only, boundary),
                value(br),
                "s",
                color=color,
                ms=5,
            )
    ax.set_xlabel("OBS optimizer updates" if exposure else "Total optimizer updates")
    ax.grid(alpha=0.2)


fig, axs = plt.subplots(2, 2, figsize=(12, 7.5))
for j, exposure in enumerate([False, True]):
    draw(axs[0, j], "blend", exposure)
    axs[0, j].set_yscale("log")
    axs[0, j].set_title("Full history · logarithmic y-axis")
    draw(axs[1, j], "blend", exposure, True)
    axs[1, j].set_title("Later training · from total update 1,500")
    for ax in axs[:, j]:
        ax.set_ylabel("Climatology-normalized RMSE blend ↓")
        ax.axhline(1, color="#777", ls=":", lw=1)
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(
    handles, labels, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 0.035)
)
fig.suptitle("Shared ERA5 versus OBS-only · EMA validation", fontsize=15)
fig.text(
    0.5,
    0.012,
    "Solid: main training. Dashed: OBS-only cooldown. Squares: cooldown start. One seed; identical nine-origin, eight-draw validation.",
    ha="center",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.09, 1, 0.96))
for ext in ["png", "pdf"]:
    fig.savefig(OUT / ("blend-curves." + ext), dpi=125)
plt.close(fig)
labels = [
    "Blend (normalized) ↓",
    "SST RMSE (°C) ↓",
    "Geostrophic velocity RMSE (m/s) ↓",
    "Geostrophic EKE RMSE (m²/s²) ↓",
    "OHC 0–700 m RMSE (GJ/m²) ↓",
    "OHC 700–2000 m RMSE (GJ/m²) ↓",
]
for exposure in [False, True]:
    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    for ax, key, label in zip(axs.flat, ["blend"] + keys, labels, strict=True):
        draw(ax, key, exposure, True)
        ax.set_title(label, fontsize=11)
    fig.suptitle(
        "Shared ERA5 versus OBS-only · components after total update 1,500", fontsize=14
    )
    fig.legend(
        handles,
        list(curves),
        loc="lower center",
        ncol=3,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.04),
    )
    fig.text(
        0.5,
        0.015,
        "Raw-field RMSE; surface leads 5/15/30 days, direct monthly OHC. Velocity/EKE derive from SSH, not the full-field X-ray velocity decodes.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    stem = "components-obs-exposure" if exposure else "components-total-updates"
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / (stem + "." + ext), dpi=125)
    plt.close(fig)
summary = {}
for name, rows in curves.items():
    summary[name] = {"best": min(rows, key=lambda r: r["blend"]), "final": rows[-1]}
(OUT / "comparison.json").write_text(
    json.dumps(
        dict(
            normalization=norm, origins=old["origins"], curves=curves, summary=summary
        ),
        indent=2,
    )
    + "\n"
)
for name, result in summary.items():
    for which, row in result.items():
        print(
            name,
            which,
            row["step"],
            round(row["blend"], 6),
            {
                k: round(row["metrics"][k] / (1e9 if k.startswith("ohc") else 1), 6)
                for k in keys
            },
        )
