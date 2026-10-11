# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Summarize saved mixed-task logs without launching model evaluations."""

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.input.read_bytes())
    event_keys = {
        "event",
        "global_step",
        "task",
        "om4",
        "observation",
        "lr",
        "loss",
        "accessory_mse",
        "physical_objective",
        "om4_retention",
    }
    for run in data.values():
        run["events"] = [
            {key: value for key, value in event.items() if key in event_keys}
            for event in run["events"]
            if event.get("event") in {"joint_train", "joint_validation"}
        ]
        for event in run["events"]:
            if event.get("task") == "observation":
                event.pop("loss", None)
        run["best"].pop("metrics", None)
        run.pop("validation_final", None)
    source = json.dumps(data, separators=(",", ":")).encode()
    args.output.mkdir(exist_ok=True, parents=True)
    (args.output / "previous-om4-evidence.json.gz").write_bytes(
        gzip.compress(source, mtime=0)
    )
    updates = {
        arm: [e for e in run["events"] if e.get("event") == "joint_train"]
        for arm, run in data.items()
    }
    validation = {
        arm: {
            e["global_step"]: e["om4_retention"]
            for e in run["events"]
            if e.get("event") == "joint_validation"
        }
        for arm, run in data.items()
    }
    contract_keys = (
        "effective_mean",
        "effective_std",
        "data_manifest_sha256",
        "grid_sha256",
        "statistics_sha256",
        "training_months",
        "validation_months",
    )
    schedule_keys = ("global_step", "task", "om4", "observation", "lr")
    reference = [tuple(e[k] for k in schedule_keys) for e in updates["control"]]
    for arm, events in updates.items():
        assert len(events) == 4000
        assert [tuple(e[k] for k in schedule_keys) for e in events] == reference
        assert sum(e["task"] == "om4" for e in events) == 2000
        for key in contract_keys:
            assert data[arm]["manifest"][key] == data["control"]["manifest"][key]
        for event in events:
            if arm != "control" and event["task"] == "om4":
                assert math.isclose(
                    event["physical_objective"] + 0.01 * event["accessory_mse"],
                    event["loss"],
                    rel_tol=1e-12,
                )

    def physical(event):
        return event.get("physical_objective", event["loss"])

    rows = []
    for start in range(1, 4001, 1000):
        values = {
            arm: [
                physical(e)
                for e in events
                if e["task"] == "om4" and start <= e["global_step"] < start + 1000
            ]
            for arm, events in updates.items()
        }
        for arm, values_arm in values.items():
            value = mean(values_arm)
            rows.append(
                dict(
                    comparison="training_block",
                    metric="physical_objective",
                    start=start,
                    end=start + 999,
                    arm=arm,
                    count=len(values_arm),
                    value=value,
                    percent_change=100 * (value / mean(values["control"]) - 1),
                )
            )
    for label, steps in (
        ("matched_3800", [3800]),
        ("matched_4000", [4000]),
        ("late_checkpoint_mean", list(range(3100, 4001, 100))),
    ):
        for metric in ("full_mse", "ts_mse", "reconstruction_ts_mse"):
            base = mean(validation["control"][s][metric] for s in steps)
            for arm in data:
                value = mean(validation[arm][s][metric] for s in steps)
                rows.append(
                    dict(
                        comparison=label,
                        metric="sqrt_mean_" + metric,
                        start=min(steps),
                        end=max(steps),
                        arm=arm,
                        count=len(steps),
                        value=math.sqrt(value),
                        percent_change=100 * (math.sqrt(value / base) - 1),
                    )
                )
    with (args.output / "previous-om4-summary.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for arm, color in zip(data, ("#333333", "#0072B2", "#D55E00"), strict=True):
        train = [
            r for r in rows if r["arm"] == arm and r["comparison"] == "training_block"
        ]
        axes[0].plot(
            [r["end"] for r in train],
            [r["percent_change"] for r in train],
            "o-",
            label=arm,
            color=color,
        )
        steps = list(range(3100, 4001, 100))
        axes[1].plot(
            steps,
            [math.sqrt(validation[arm][s]["full_mse"]) for s in steps],
            "o-",
            label=arm,
            color=color,
        )
    axes[0].set(
        title="OM4 training physical objective",
        xlabel="End of 1,000-update block",
        ylabel="Change versus control (%)",
    )
    axes[0].axhline(0, color="gray", linewidth=0.6)
    axes[1].set(
        title="OM4 validation: 11 fixed origins",
        xlabel="Total training updates",
        ylabel="Variable-balanced normalized forecast RMSE",
    )
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend()
    fig.savefig(args.output / "previous-om4-comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"rows": len(rows), "matched_updates": 4000, "om4_updates": 2000}))


if __name__ == "__main__":
    main()
