# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Derive per-level skill and physical velocity moments from collected CSVs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib
from artifact_io import artifact_exists, open_artifact

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
MODES = ("inferred", "true", "inferred_persistence", "climatology")
REGIONS = ("global", "tropics", "extratropics")
LEADS = (5, 10, 15, 20, 25, 30)
STYLES = {
    "inferred": ("#2166ac", "-", "Inferred interior + evolution"),
    "true": ("#1b9e77", "--", "True interior + evolution"),
    "inferred_persistence": ("#d95f02", ":", "Inferred-state persistence"),
    "climatology": ("#777777", "-.", "Monthly climatology"),
}


def write_csv(name, rows):
    with (ROOT / name).open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


level_rows, moment_rows = [], []
summaries: dict[str, dict[str, dict[str, int]]] = {}
for task in ("ar", "direct"):
    path = ROOT / task / "heldout_metrics.csv"
    if not artifact_exists(path):
        continue
    manifest = json.loads((path.parent / "manifest.json").read_text())
    assert (path.parent / "COMPLETE.json").exists()
    assert manifest["arguments"]["max_steps"] == 0
    rows = list(csv.DictReader(open_artifact(path)))
    lookup = {
        (r["mode"], r["region"], int(r["lead_days"]), r["channel"]): r for r in rows
    }
    expected = {
        (mode, region, lead, channel)
        for mode in MODES
        for region in REGIONS
        for lead in LEADS
        for channel in manifest["channels"]
    }
    assert set(lookup) == expected and len(lookup) == len(rows) == 5544
    assert {int(r["origins"]) for r in rows} == {99}
    for row in rows:
        for field in (
            "normalized_rmse",
            "physical_rmse",
            "prediction_second_moment",
            "target_second_moment",
        ):
            assert math.isfinite(float(row[field])) and float(row[field]) >= 0
    summaries[task] = {}
    for region in REGIONS:
        for lead in LEADS:
            for channel in manifest["channels"]:
                inferred = lookup["inferred", region, lead, channel]
                physical = {
                    mode: float(lookup[mode, region, lead, channel]["physical_rmse"])
                    for mode in MODES
                }
                variable = "sst" if channel == "thetao_0" else channel.split("_")[0]
                depth = (
                    ""
                    if channel == "zos"
                    else manifest["depths_m"][int(channel.split("_")[1])]
                )
                out = dict(
                    model=task,
                    region=region,
                    lead_days=lead,
                    channel=channel,
                    variable=variable,
                    depth_m=depth,
                    origins=99,
                )
                out.update(
                    {f"{mode}_physical_rmse": value for mode, value in physical.items()}
                )
                for baseline in ("inferred_persistence", "climatology"):
                    out[f"rmse_improvement_over_{baseline}_pct"] = (
                        100 * (1 - physical["inferred"] / physical[baseline])
                        if physical[baseline]
                        else ""
                    )
                level_rows.append(out)
            for mode in MODES:
                # Equal level weighting, each component already area averaged over its wet cells.
                # Raw physical second moments include mean flow; this is not eddy energy.
                channels = [
                    c for c in manifest["channels"] if c.startswith(("uo_", "vo_"))
                ]
                p2 = sum(
                    float(lookup[mode, region, lead, c]["prediction_second_moment"])
                    for c in channels
                )
                t2 = sum(
                    float(lookup[mode, region, lead, c]["target_second_moment"])
                    for c in channels
                )
                assert t2 > 0
                moment_rows.append(
                    dict(
                        model=task,
                        mode=mode,
                        region=region,
                        lead_days=lead,
                        predicted_mean_component_second_moment=p2 / len(channels),
                        target_mean_component_second_moment=t2 / len(channels),
                        second_moment_ratio=p2 / t2,
                        rms_amplitude_ratio=math.sqrt(p2 / t2),
                        levels=19,
                        origins=99,
                    )
                )
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    for ax, (variable, label, unit) in zip(
        axes,
        [
            ("thetao", "Subsurface temperature", "degrees C"),
            ("so", "Salinity", "native salinity units"),
        ],
    ):
        indices = list(range(1 if variable == "thetao" else 0, 19))
        depths = [manifest["depths_m"][i] for i in indices]
        for mode in MODES:
            color, style, title = STYLES[mode]
            errors = [
                float(lookup[mode, "global", 30, f"{variable}_{i}"]["physical_rmse"])
                for i in indices
            ]
            ax.plot(
                errors,
                depths,
                color=color,
                linestyle=style,
                marker="o",
                markersize=3,
                label=title,
            )
        ax.set_yscale("log")
        ax.invert_yaxis()
        ax.set_title(label)
        ax.set_xlabel(f"Physical RMSE ({unit})")
        ax.set_ylabel("Depth (m, logarithmic scale)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(
        f"{task.upper()}: 30-day errors by depth, 99 OM4 held-out starts\nGlobal wet-cell area-weighted errors; five-day data",
        fontsize=12,
    )
    fig.savefig(ROOT / f"{task}_depth_rmse.png", dpi=180)
    fig.savefig(ROOT / f"{task}_depth_rmse.svg")
    plt.close(fig)
    for region in REGIONS:
        for variable in ("thetao", "so"):
            subset = [
                r
                for r in level_rows
                if r["model"] == task
                and r["region"] == region
                and r["lead_days"] == 30
                and r["variable"] == variable
            ]
            summaries[task][f"{region}/{variable}/30"] = {
                "levels": len(subset),
                "levels_beating_persistence": sum(
                    r["rmse_improvement_over_inferred_persistence_pct"] > 0
                    for r in subset
                ),
                "levels_beating_climatology": sum(
                    r["rmse_improvement_over_climatology_pct"] > 0 for r in subset
                ),
            }
assert level_rows and moment_rows
write_csv("depth_skill.csv", level_rows)
write_csv("velocity_moments.csv", moment_rows)
(ROOT / "depth_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
print(json.dumps(summaries, indent=2))
print("Global inferred 30-day physical velocity moments:")
print(
    json.dumps(
        [
            r
            for r in moment_rows
            if r["mode"] == "inferred"
            and r["region"] == "global"
            and r["lead_days"] == 30
        ],
        indent=2,
    )
)
