#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Render a late-training view and matched maps from saved early/fine results."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from plot_observation_rollout_maps import digest, plot


def render(final, original_maps, early_maps):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = final / "summary"
    with (summary / "validation-curves.csv").open() as stream:
        curves = list(csv.DictReader(stream))
    with (summary / "scores.csv").open() as stream:
        scores = [r for r in csv.DictReader(stream) if r["forecast"] == "evolved"]
    sources = json.loads((final / "source-results.json").read_text())["arms"]
    sources.update(json.loads((final / "u-global.json").read_text())["arms"])
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4.8), sharey=True, constrained_layout=True
    )
    for ax, key, xlabel in zip(
        axes,
        ["optimizer_updates", "observation"],
        ["All optimizer updates", "Observation updates"],
    ):
        for row in scores:
            name = row["arm"]
            points = [
                p
                for p in curves
                if p["arm"] == name and int(p["optimizer_updates"]) >= 2000
            ]
            line = ax.plot(
                [int(p[key]) for p in points],
                [float(p["score"]) for p in points],
                label=name,
            )[0]
            best = sources[name]["best"]
            x = (
                best["global_step"]
                if key == "optimizer_updates"
                else best["task_counts"]["observation"]
            )
            ax.scatter(
                x, best["score"], marker="*", s=90, color=line.get_color(), zorder=5
            )
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Observation validation composite (lower is better)")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Final 2,000 updates — one seed; stars mark selected checkpoints")
    fig.savefig(summary / "late-validation.png", dpi=160)
    plt.close(fig)

    with np.load(original_maps) as data:
        old = {k: data[k] for k in data.files}
    with np.load(early_maps) as data:
        new = {k: data[k] for k in data.files}
    for key in ("lat", "lon", "mask", "reference", "climatology"):
        np.testing.assert_array_equal(old[key], new[key])
    op, npv = [json.loads(str(d["provenance"])) for d in (old, new)]
    if op["case_times"] != npv["case_times"] or op["grid_sha256"] != npv["grid_sha256"]:
        raise ValueError("Map cases or grid differ")
    old_names = [s["method"] for s in op["sources"]]
    baseline = old_names.index("U-global")
    persistence = old_names.index("U-global / initialized persistence")
    indices = [0, 2, 4, 6]
    new["prediction"] = np.stack(
        [
            old["prediction"][baseline],
            *[new["prediction"][i] for i in indices],
            old["prediction"][persistence],
            new["prediction"][7],
        ]
    )
    control = op["sources"][persistence]
    npv["sources"] = [
        op["sources"][baseline],
        *[npv["sources"][i] for i in indices],
        control,
        npv["sources"][7],
        npv["sources"][-1],
    ]
    npv["input_bundles"] = {str(p): digest(p) for p in (original_maps, early_maps)}
    npv["merge_script_sha256"] = digest(Path(__file__))
    npv["surface_persistence_note"] = (
        "Initialized persistence remains model-specific, including surface values filled where initial observations are unavailable."
    )
    new["provenance"] = np.array(json.dumps(npv))
    bundle = final / "maps-combined.npz"
    np.savez_compressed(bundle, **new)
    plot(bundle, final / "maps", global_observations=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--original-maps", type=Path, required=True)
    parser.add_argument("--early-maps", type=Path, required=True)
    args = parser.parse_args()
    render(args.final, args.original_maps, args.early_maps)
