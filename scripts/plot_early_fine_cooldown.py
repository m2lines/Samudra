#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot matched cooldown curves and fixed-case maps from verified saved results."""

import argparse
import json
from pathlib import Path

import numpy as np
from plot_observation_rollout_maps import digest, plot


def render(final, parent, cooldown_maps):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cooled = json.loads((final / "source-results.json").read_text())["arms"]
    parents = json.loads((parent / "source-results.json").read_text())["arms"]
    parents.update(json.loads((parent / "u-global.json").read_text())["arms"])
    names = ["U-global", *[n for n in parents if n != "U-global"]]
    for late in (False, True):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        for ax, name in zip(axes.flat, names):
            for label, record, color in (
                ("Constant rate", parents[name], "#757575"),
                ("Cooldown", cooled[name + "-cooldown"], "#0072B2"),
            ):
                points = record["validation_events"]
                points = [p for p in points if not late or p["global_step"] >= 2400]
                ax.plot(
                    [p["global_step"] for p in points],
                    [p["validation/obs_score"] for p in points],
                    label=label,
                    color=color,
                )
                best = record["best"]
                ax.scatter(
                    best["global_step"], best["score"], marker="*", s=95, color=color
                )
            ax.axvline(2667, color="black", ls=":", alpha=0.4, label="Fork")
            ax.axvline(3000, color="black", ls="--", alpha=0.4, label="Decay begins")
            ax.set(
                title=name,
                xlabel="All optimizer updates",
                ylabel="Validation composite",
            )
            ax.grid(alpha=0.2)
        axes.flat[-1].axis("off")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        axes.flat[-1].legend(handles, labels, loc="center", frameon=False)
        fig.suptitle(
            "Matched learning-rate cooldown — one seed; lower is better; stars select checkpoints"
        )
        stem = "validation-late" if late else "validation-full"
        fig.savefig(final / (stem + ".png"), dpi=160)
        fig.savefig(final / (stem + ".pdf"))
        plt.close(fig)

    with np.load(parent / "maps-combined.npz") as data:
        old = {k: data[k] for k in data.files}
    with np.load(cooldown_maps) as data:
        new = {k: data[k] for k in data.files}
    for key in ("lat", "lon", "mask", "reference", "climatology"):
        np.testing.assert_array_equal(old[key], new[key])
    op, npv = [json.loads(str(d["provenance"])) for d in (old, new)]
    if op["case_times"] != npv["case_times"] or op["grid_sha256"] != npv["grid_sha256"]:
        raise ValueError("Map cases or grid differ")
    old_names = [s["method"] for s in op["sources"]]
    new_names = [s["method"] for s in npv["sources"]]
    for name in names:
        predictions, sources = [], []
        for bundle, provenance, labels, label in (
            (old, op, old_names, name),
            (new, npv, new_names, name + "-cooldown"),
            (new, npv, new_names, name + "-cooldown / initialized persistence"),
        ):
            index = labels.index(label)
            predictions.append(bundle["prediction"][index])
            sources.append(provenance["sources"][index])
        sources.append(npv["sources"][-1])
        provenance = dict(
            npv,
            sources=sources,
            input_bundles={
                str(p): digest(p) for p in (parent / "maps-combined.npz", cooldown_maps)
            },
            merge_script_sha256=digest(Path(__file__)),
        )
        arrays = dict(new)
        arrays["prediction"] = np.stack(predictions)
        arrays["provenance"] = np.array(json.dumps(provenance))
        folder = final / "maps" / name
        folder.mkdir(parents=True, exist_ok=True)
        output = folder / "source.npz"
        np.savez_compressed(output, **arrays)
        plot(output, folder, global_observations=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--cooldown-maps", type=Path, required=True)
    args = parser.parse_args()
    render(args.final, args.parent, args.cooldown_maps)
