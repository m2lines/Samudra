#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Plot saved full-precision validation; does not run model inference."""

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--evaluation", type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1] / "docs/experiments/observation-d"
    baseline = json.loads(
        (root / "assets/shared-era5-2026-10-26/metrics.json").read_text()
    )
    control = json.loads(
        (root / "assets/monthly-replay-2026-10-20/results.json").read_text()
    )
    norm = control["normalization"]
    result = json.loads(args.evaluation.read_text())
    rows = []
    for source in ("shared8k", "shared4k"):
        for filename, payload in baseline[source].items():
            match = re.fullmatch(r"validation-(raw-)?(\d+)\.json", filename)
            if not match:
                continue
            step = int(match[2])
            if (
                step % 500
                or (source == "shared8k" and step > 4000)
                or (source == "shared4k" and step <= 4000)
            ):
                continue
            assert payload["origins"] == control["origins"]
            assert payload["members"] == 8 and payload["sampler_steps"] == 32
            metrics = {k: payload["metrics"][k] for k in norm}
            rows.append(
                dict(
                    model="Baseline 0.8/0.2/0.1",
                    weights="raw" if match[1] else "ema",
                    step=step,
                    blend=float(np.mean([metrics[k] / norm[k] for k in norm])),
                    **metrics,
                )
            )
    for receipt in result["checkpoints"]:
        assert receipt["origins"] == control["origins"]
        assert receipt["members"] == 8 and receipt["sampler_steps"] == 32
        for weights, data in receipt["weights"].items():
            metrics = {k: data["metrics"][k] for k in norm}
            assert (
                abs(np.mean([metrics[k] / norm[k] for k in norm]) - data["blend"])
                < 1e-10
            )
            rows.append(
                dict(
                    model="OBS balanced 0.5/0.5/0.1",
                    weights=weights,
                    step=receipt["step"],
                    blend=data["blend"],
                    **metrics,
                )
            )
    out = root / "assets/shared-era5-obs-reweight-2026-10-26"
    out.mkdir(exist_ok=True)
    with (out / "curves.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.7))
    titles = [
        "RMSE blend (lower is better)",
        "SST RMSE (°C)",
        "Velocity RMSE (m/s)",
        "EKE RMSE (m²/s²)",
        "OHC 0–700 m (GJ/m²)",
        "OHC 700–2000 m (GJ/m²)",
    ]
    for ax, key, title in zip(axes.flat, ["blend", *norm], titles, strict=True):
        for model, color in [
            ("Baseline 0.8/0.2/0.1", "#0072B2"),
            ("OBS balanced 0.5/0.5/0.1", "#D55E00"),
        ]:
            for weights, style in [("ema", "-"), ("raw", ":")]:
                curve = sorted(
                    [
                        r
                        for r in rows
                        if r["model"] == model and r["weights"] == weights
                    ],
                    key=lambda r: r["step"],
                )
                ax.plot(
                    [r["step"] for r in curve],
                    [r[key] / (1e9 if key.startswith("ohc") else 1) for r in curve],
                    style,
                    color=color,
                    marker=".",
                    alpha=1 if weights == "ema" else 0.5,
                    label=model + " " + weights.upper(),
                )
        ax.axvline(4000, color="gray", ls="--", lw=0.8)
        ax.set(title=title, xlabel="Total updates")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(
        "Shared ERA5 loss reweighting · seed 1729 · cooldown starts at 4k\nSame full-precision validation; baseline raw checkpoints are sparse"
    )
    fig.tight_layout()
    fig.savefig(out / "validation-curves.png", dpi=120)
    for name in ("curves.csv", "validation-curves.png"):
        (out / (name + ".license")).write_text(
            "SPDX-FileCopyrightText: 2026 Samudra Authors\nSPDX-License-Identifier: CC-BY-4.0\n"
        )
    for model in sorted({r["model"] for r in rows}):
        selected = min(
            (r for r in rows if r["model"] == model and r["weights"] == "ema"),
            key=lambda r: r["blend"],
        )
        print(json.dumps(selected))


if __name__ == "__main__":
    main()
