# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Undetrended annual supported means and heat totals for stopped latent training."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from global_physical_heat_content import (  # type: ignore[import-not-found]
    cell_areas,
    heat_total,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--short-runs", type=Path, required=True)
    p.add_argument("--stopped-runs", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--heat-context", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    grid = dict(np.load(a.references / "grid.npz"))
    lat, lon = grid["lat"], grid["lon"]
    area = cell_areas(lat, lon)
    cosine = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, len(lon)))
    heat_climate = np.load(a.heat_context)["climatology"]
    output = {}
    styles = [
        ("Deterministic historical", "black", "^"),
        ("Short 1729", "#cd5c5c", "D"),
        ("Short 1730", "#c76e00", "X"),
        ("Stopped 1729", "#168445", "P"),
        ("Stopped 1730", "#206cb0", "v"),
    ]
    for year in (2015, 2018, 2021):
        origin = f"{year}-01-01"
        paths = [a.baseline / "annual" / f"{origin}.npz"] + [
            root / f"D-{seed}" / "report-v1/annual" / f"{origin}.npz"
            for root in (a.short_runs, a.stopped_runs)
            for seed in (1729, 1730)
        ]
        models = [dict(np.load(path)) for path in paths]
        for model in models[1:]:
            np.testing.assert_allclose(
                model["reference"], models[0]["reference"], equal_nan=True
            )
            np.testing.assert_allclose(
                model["reference_ohc"], models[0]["reference_ohc"], equal_nan=True
            )
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
        records = {}
        for panel, (label, unit) in enumerate(
            [("SST", "°C"), ("SSH", "m"), ("OHC 0–700m", "ZJ"), ("OHC 700–2000m", "ZJ")]
        ):
            ax = axes.flat[panel]
            channel = panel % 2
            is_heat = panel >= 2
            truth = (
                models[0]["reference_ohc"][:, channel]
                if is_heat
                else models[0]["reference"][:, channel]
            )
            values = [
                m["predicted_ohc"][:, channel] if is_heat else m["surface"][:, channel]
                for m in models
            ]
            wet = grid["mask"][0 if channel == 0 else 6].astype(bool)
            support = wet & np.isfinite(truth).all(0)
            for field in values:
                support &= np.isfinite(field).all(0)
            if is_heat:
                climate = heat_climate[:, channel]
                support &= np.isfinite(climate).all(0)
            weights = area if is_heat else cosine

            def reduce(field):
                if is_heat:
                    return heat_total(field, area, support)
                return (
                    np.sum(np.where(support, field, 0) * weights, axis=(-2, -1))
                    / weights[support].sum()
                )

            days = np.arange(1, 13) if is_heat else 5 * np.arange(1, 74)
            series = {"Observations": reduce(truth).tolist()}
            ax.plot(
                days,
                series["Observations"],
                color="#444444",
                marker="o",
                markevery=12 if not is_heat else 2,
                label="Observations",
            )
            for field, (name, color, marker) in zip(values, styles, strict=True):
                series[name] = reduce(field).tolist()
                ax.plot(
                    days,
                    series[name],
                    color=color,
                    marker=marker,
                    markevery=12 if not is_heat else 2,
                    label=name,
                )
            if is_heat:
                series["Training climatology"] = reduce(climate).tolist()
                ax.plot(
                    days,
                    series["Training climatology"],
                    color="#9068a8",
                    marker="s",
                    markevery=2,
                    label="Training climatology",
                )
            ax.set(
                title=label, ylabel=unit, xlabel="Month" if is_heat else "Lead (days)"
            )
            ax.grid(alpha=0.2)
            records[label] = dict(
                series=series,
                x=days.tolist(),
                units=unit,
                support_area_m2=float(area[support].sum()),
                wet_area_fraction=float(area[support].sum() / area[wet].sum()),
            )
        handles, labels = axes.flat[3].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=9)
        fig.suptitle(
            f"{origin}: one initialization; undetrended means / supported heat totals"
        )
        fig.savefig(a.output / f"annual-means-{year}.png", dpi=130)
        plt.close(fig)
        output[origin] = records
    (a.output / "annual-means.json").write_text(
        json.dumps(
            dict(
                scope="Common fixed finite support per year and variable; globally distributed support, no extrapolation to missing ocean. OHC model/native vertical discretizations differ.",
                results=output,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
