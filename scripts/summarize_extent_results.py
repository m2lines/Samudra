#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize verified extent results with common controls and actual update axes."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


def actual_updates(arm, event):
    if "optimizer_updates" in event:
        return event["optimizer_updates"]
    # Compatibility with the first completed omission report, which predates
    # explicit validation-event counters. Its frozen schedule omits every
    # second OM4 slot, as independently checked in EXPOSURE.json.
    omitted = arm.get("exposure", {}).get("omitted_patch_slots", 0)
    return event["global_step"] - (event["om4"] // 2 if omitted else 0)


def controls_match(left, right, rtol):
    """Allow an explicit relative roundoff tolerance, preserving raw inputs."""
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            controls_match(left[key], right[key], rtol) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            controls_match(a, b, rtol) for a, b in zip(left, right)
        )
    if isinstance(left, float) and isinstance(right, float):
        return math.isclose(left, right, rel_tol=rtol, abs_tol=0)
    return left == right


def summarize(sources, output, groups, control_rtol=0):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bundles = [json.loads(path.read_text()) for path in sources]
    if len({b["selection_reference_sha256"] for b in bundles}) != 1:
        raise ValueError("Validation references differ")
    arms = {}
    for bundle in bundles:
        for name, arm in bundle["arms"].items():
            if name in arms:
                raise ValueError("Duplicate model name in source bundles")
            arms[name] = arm
    reference = bundles[0]["selection_reference"]
    integrated = reference["protocol"]["integrated"]
    spectral = reference["spectral_keys"]
    control = next(iter(arms.values()))["files"]["seasonal-climatology.json"]["data"]
    scores, curves = [], []
    for name, arm in arms.items():
        if not controls_match(
            arm["files"]["seasonal-climatology.json"]["data"], control, control_rtol
        ):
            raise ValueError("Held-out normalization controls differ")
        best = arm["best"]
        counts = best["task_counts"]
        selected_actual = actual_updates(
            arm, dict(global_step=best["global_step"], om4=counts["om4"])
        )
        for label, filename in [
            ("evolved", "selected.json"),
            ("initialized persistence", "selected-inferred-persistence.json"),
            (
                "initialized anomaly persistence",
                "selected-inferred-anomaly-persistence.json",
            ),
        ]:
            metrics = arm["files"][filename]["data"]
            if label == "evolved":
                metrics = metrics["metrics"]
            if len(metrics["origins"]) != 96:
                raise ValueError("Incomplete test cohort")
            ratio = statistics.mean(
                metrics["metrics"][key] / control["metrics"][key] for key in integrated
            )
            dex = statistics.mean(
                metrics["spectra"][key]["error_dex"] for key in spectral
            )
            scores.append(
                dict(
                    arm=name,
                    forecast=label,
                    selected_schedule_slot=best["global_step"],
                    selected_optimizer_updates=selected_actual,
                    validation_score=best["score"],
                    composite=(ratio + dex) / 2,
                    integrated_ratio=ratio,
                    spectral_dex=dex,
                    **{k: metrics["metrics"][k] for k in integrated},
                )
            )
        for event in arm["validation_events"]:
            curves.append(
                dict(
                    arm=name,
                    schedule_slot=event["global_step"],
                    optimizer_updates=actual_updates(arm, event),
                    scheduled_om4=event["om4"],
                    observation=event["observation"],
                    score=event["validation/obs_score"],
                )
            )
    output.mkdir(parents=True, exist_ok=True)
    for filename, rows in [("scores.csv", scores), ("validation-curves.csv", curves)]:
        with (output / filename).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    rendered = []
    for group, names in groups.items():
        if not set(names) <= arms.keys():
            raise ValueError("Plot group includes an unavailable model")
        fig, axes = plt.subplots(
            1, 2, figsize=(12, 4.8), sharey=True, constrained_layout=True
        )
        for axis, key, xlabel in zip(
            axes,
            ["optimizer_updates", "observation"],
            ["Actual optimizer updates (all tasks)", "Observation updates"],
        ):
            for name in names:
                points = [x for x in curves if x["arm"] == name]
                line = axis.plot(
                    [x[key] for x in points],
                    [x["score"] for x in points],
                    label=name,
                    linewidth=1.7,
                )[0]
                selected = next(
                    x for x in scores if x["arm"] == name and x["forecast"] == "evolved"
                )
                x = (
                    selected["selected_optimizer_updates"]
                    if key == "optimizer_updates"
                    else arms[name]["best"]["task_counts"]["observation"]
                )
                axis.scatter(
                    x,
                    selected["validation_score"],
                    marker="*",
                    color=line.get_color(),
                    s=60,
                    zorder=3,
                )
            axis.set_xlabel(xlabel)
            axis.grid(alpha=0.2)
            axis.legend(frameon=False, fontsize=8)
            axis.set_xlim(left=0)
        axes[0].set_ylabel("Observation validation composite (lower is better)")
        fig.suptitle(group + " — one seed; stars mark selected checkpoints")
        filename = group.lower().replace(" ", "-") + ".png"
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)
        (output / (filename + ".license")).write_text(
            "SPDX-FileCopyrightText: 2026 Samudra Authors\nSPDX-License-Identifier: CC-BY-4.0\n"
        )
        rendered.append(filename)
    provenance = dict(
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sources={
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
        },
        selection_reference_sha256=bundles[0]["selection_reference_sha256"],
        test_normalization="Common held-out seasonal-climatology errors from the first source arm; all controls verified within the recorded relative tolerance. Validation uses its separately frozen reference.",
        control_relative_tolerance=control_rtol,
        x_axes="Actual optimizer updates include all trained tasks and exclude omitted slots; observation updates show equal downstream exposure separately.",
        figures=rendered,
    )
    (output / "summary-provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control-rtol", type=float, default=0)
    parser.add_argument(
        "--groups",
        type=Path,
        required=True,
        help="JSON mapping figure title to literal model names",
    )
    args = parser.parse_args()
    summarize(
        args.sources,
        args.output,
        json.loads(args.groups.read_text()),
        args.control_rtol,
    )


if __name__ == "__main__":
    main()
