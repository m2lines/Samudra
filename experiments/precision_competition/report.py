# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Export complete and censored runs, matched-step penalties, and learning curves."""

import argparse
import csv
import gzip
import json
import math
from dataclasses import replace
from pathlib import Path

from .benchmark import Config, atomic_json


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(manifest, source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    rows, curves, pairs, thresholds, records = [], [], [], [], []
    histories = {}
    for entry in manifest:
        config = Config(**entry["config"])
        root = source / config.run_id
        result_path, history_path = root / "result.json", root / "history.json"
        history = json.loads(history_path.read_text()) if history_path.exists() else []
        histories[config.run_id] = {row["step"]: row for row in history}
        result = json.loads(result_path.read_text()) if result_path.exists() else None
        records.append({"entry": entry, "history": history, "result": result})
        row = {
            "id": config.run_id,
            "families": ";".join(entry["families"]),
            **entry["config"],
            "status": "complete" if result else "incomplete",
        }
        if result:
            row.update(
                endpoint_step=result["step"],
                stop_reason=result["stop_reason"],
                training_seconds=result["training_seconds"],
                parameters=result["provenance"]["parameters"],
                device=result["provenance"]["device"],
            )
            row.update(
                {
                    "test_" + k: v
                    for k, v in result["test"].items()
                    if not isinstance(v, dict)
                }
            )
            row.update(
                {
                    "spectral_" + k: v
                    for k, v in result["spectrum"].items()
                    if not isinstance(v, list)
                }
            )
        rows.append(row)
        for item in history:
            curves.append(
                {
                    "id": config.run_id,
                    **{k: v for k, v in item.items() if not isinstance(v, dict)},
                }
            )
        for bits in (4, 8, 12, 16, 20):
            streak, reached = 0, None
            for item in history:
                streak = streak + 1 if item["normalized_rmse"] <= 2**-bits else 0
                if streak == 3:
                    reached = item
                    break
            thresholds.append(
                {
                    "id": config.run_id,
                    "bits": bits,
                    "reached": reached is not None,
                    "incomplete": result is None,
                    "sustained_step": reached["step"] if reached else None,
                    "training_seconds": reached["training_seconds"]
                    if reached
                    else None,
                    "approximate_training_flops": reached["approximate_training_flops"]
                    if reached
                    else None,
                    "last_observed_step": history[-1]["step"] if history else None,
                }
            )
    for entry in manifest:
        config = Config(**entry["config"])
        if config.auxiliary_weight == 0 or config.main_weight == 0:
            continue
        solo = replace(
            config, auxiliary_weight=0.0, relation="independent", auxiliary_start=0
        )
        a, b = histories.get(solo.run_id, {}), histories.get(config.run_id, {})
        common = sorted(set(a) & set(b))
        for step in sorted(set(common) | {100, 500, 1000, 2000, 3000}):
            available = step in a and step in b
            row = {
                "id": config.run_id,
                "solo_id": solo.run_id,
                "step": step,
                "available": available,
                "width": config.width,
                "depth": config.depth,
                "heads": config.heads,
                "seed": config.seed,
                "main": config.main,
                "objective": config.objective,
                "relation": config.relation,
                "auxiliary_weight": config.auxiliary_weight,
            }
            if available:
                error_a, error_b = a[step]["rmse"], b[step]["rmse"]
                row.update(
                    bits_lost=math.log2(error_b / error_a)
                    if error_a and error_b
                    else None,
                    solo_rmse=error_a,
                    joint_rmse=error_b,
                    auxiliary_rmse=b[step]["auxiliary_rmse"],
                    solo_training_seconds=a[step]["training_seconds"],
                    joint_training_seconds=b[step]["training_seconds"],
                    joint_approximate_training_flops=b[step][
                        "approximate_training_flops"
                    ],
                )
            pairs.append(row)
    write_csv(destination / "runs.csv", rows)
    write_csv(destination / "curves.csv", curves)
    write_csv(destination / "bits-lost.csv", pairs)
    write_csv(destination / "thresholds.csv", thresholds)
    with gzip.open(destination / "records.json.gz", "wt") as stream:
        json.dump(records, stream, allow_nan=False)
    summary = {
        "planned": len(rows),
        "complete": sum(r["status"] == "complete" for r in rows),
        "training_gpu_hours_completed": sum(r.get("training_seconds", 0) for r in rows)
        / 3600,
        "note": "Training time excludes validation, startup, checkpoints, preemption replay and scheduler overhead; use sacct for allocated cost.",
    }
    atomic_json(destination / "summary.json", summary)
    plot(records, destination)
    return summary


def plot(records, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for record in records:
        c = Config(**record["entry"]["config"])
        if (
            not record["history"]
            or c.seed != 11
            or c.run_id
            not in {
                replace(Config(), auxiliary_weight=w, relation=r).run_id
                for w, r in [
                    (0.0, "independent"),
                    (1.0, "independent"),
                    (1.0, "dependent"),
                ]
            }
        ):
            continue
        h = record["history"]
        label = "solo" if not c.auxiliary_weight else c.relation
        axes[0].plot([x["step"] for x in h], [x["bits"] for x in h], label=label)
        axes[1].plot(
            [x["training_seconds"] for x in h], [x["bits"] for x in h], label=label
        )
    for ax, xlabel in zip(
        axes, ["Optimizer updates", "Measured training seconds"], strict=True
    ):
        ax.set(xlabel=xlabel, ylabel="Effective RMS bits; higher is better")
        ax.grid(alpha=0.25)
        if ax.lines:
            ax.legend()
    fig.suptitle("Initial copy screen: width 128, depth 4, 2 heads; seed 11 only")
    fig.savefig(destination / "copy-curves.png", dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            summarize(json.loads(args.manifest.read_text()), args.source, args.output)
        )
    )


if __name__ == "__main__":
    main()
