# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""CPU raw/EMA curves for the large classifier and original matched regressor."""

import argparse
import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "curve_helpers", HERE.parent / "soft-priors-25k/render_curves.py"
)
assert SPEC is not None and SPEC.loader is not None
h = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(h)
REFERENCE = Path(
    "/home/jder/.local/state/codex-async-followup/workflows/"
    "skyline-comparison-artifacts/all_fields-25k"
)


def load_runs(classifier, regression):
    runs = [
        h.read_run(classifier, "large-classifier"),
        h.read_run(regression, "original-regression", benchmark=True),
    ]
    for run, objective, label in zip(
        runs,
        ("absolute_ce", "regression"),
        ("Large classifier", "Original all-fields regressor"),
        strict=True,
    ):
        cfg = run["provenance"]["config"]
        if (
            cfg.get("qualification", False)
            or cfg["arm"] != "all_fields"
            or cfg["model"]["objective"] != objective
        ):
            raise ValueError(
                "Expected production all-fields classifier/regression runs"
            )
        for key, expected in {
            "seed": 20261106,
            "crop": 33,
            "validation_dates": 8,
            "validation_points_per_depth": 64,
        }.items():
            h.same(cfg[key], expected, key)
        if any(len(v["validation"]["per_channel"]) != 77 for v in run["vals"]):
            raise ValueError("All 77 validation channels are required")
        run.update(
            label=label,
            matched_validation_samples=True,
            likelihood_comparable=objective == "absolute_ce",
        )
    h.same(
        runs[0]["signature"],
        runs[1]["signature"],
        "validation dates, counts, masks and persistence",
    )
    for key in ("mean", "std", "metadata_sha256"):
        h.same(
            runs[0]["provenance"]["source"][key],
            runs[1]["provenance"]["source"][key],
            key,
        )
    for run in runs:
        for record in run["vals"]:
            h.same(
                record["validation"]["metric"],
                runs[0]["vals"][0]["validation"]["metric"],
                "metric definition",
            )
    for key in (
        "seed",
        "validation_seed",
        "microbatch",
        "accumulation",
        "world_size",
        "samples_per_frame",
        "coverage",
        "ema",
        "learning_rate",
        "warmup",
    ):
        h.same(
            runs[0]["provenance"]["config"][key],
            runs[1]["provenance"]["config"][key],
            "recipe " + key,
        )
    for key in ("width", "layers", "heads", "field_queries"):
        h.same(
            runs[0]["provenance"]["config"]["model"][key],
            runs[1]["provenance"]["config"]["model"][key],
            "backbone " + key,
        )
    return runs


def tables(runs):
    channels, fields, training = h.tables(runs)
    lookup = {run["name"]: run for run in runs}
    units = {"uo": "m/s", "vo": "m/s", "thetao": "degC", "so": "PSU", "zos": "m"}
    for row in channels:
        row["physical_units"] = units[row["field"]]
        std = lookup[row["run"]]["provenance"]["source"]["std"][row["channel_index"]]
        row["full_grid_physical_mse"] = row["full_grid_nmse"] * std**2
        row["full_grid_physical_rmse"] = float(np.sqrt(row["full_grid_physical_mse"]))
        row["persistence_full_grid_physical_mse"] = (
            row["persistence_full_grid_nmse"] * std**2
        )
        row["persistence_full_grid_physical_rmse"] = float(
            np.sqrt(row["persistence_full_grid_physical_mse"])
        )
    for row in fields:
        row["physical_units"] = units[row["field"]]
        selected = [
            r
            for r in channels
            if all(r[k] == row[k] for k in ("run", "step", "weights", "field"))
        ]
        for prefix in ("", "persistence_"):
            row[prefix + "full_grid_physical_rmse"] = float(
                np.sqrt(
                    np.mean([r[prefix + "full_grid_physical_mse"] for r in selected])
                )
            )
    return channels, fields, training


def plot(rows, runs, metric, panels, output):
    fig, axes = h.plt.subplots(
        int(np.ceil(len(panels) / 4)),
        4,
        figsize=(17, 3 * int(np.ceil(len(panels) / 4))),
        squeeze=False,
    )
    for ax, (name, key, value) in zip(axes.flat, panels, strict=False):
        selected = [r for r in rows if r[key] == value]
        for i, run in enumerate(runs):
            for weights, style in (("raw", "--"), ("ema", "-")):
                series = sorted(
                    (
                        r
                        for r in selected
                        if r["run"] == run["name"]
                        and r["weights"] == weights
                        and r["step"] >= 1
                        and metric in r
                    ),
                    key=lambda r: r["step"],
                )
                if series:
                    ax.plot(
                        [r["step"] for r in series],
                        [r[metric] for r in series],
                        style,
                        color=h.COLORS[i],
                        label=f"{run['label']} {weights}",
                    )
        persistence = "persistence_" + metric
        if persistence in selected[0]:
            ax.axhline(
                selected[0][persistence], color="black", ls=":", label="Persistence"
            )
        ax.set(xscale="log", xlabel="Update", ylabel=metric, title=name)
        if "nll" not in metric:
            ax.set_yscale("log")
        ax.grid(alpha=0.2)
    for ax in list(axes.flat)[len(panels) :]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(
        "Matched eight-date validation · raw dashed / EMA solid · step 0 retained in CSV"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=120)
    h.plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--regression", type=Path, default=REFERENCE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = load_runs(args.classifier, args.regression)
    channels, fields, training = tables(runs)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("validation-channels.csv.gz", channels),
        ("validation-fields.csv", fields),
        ("training-progress.csv.gz", training),
    ):
        h.write_csv(args.output / name, rows)
    panels = [(f, "field", f) for f in ("uo", "vo", "thetao", "so", "zos")]
    for metric in ("full_grid_nmse", "full_grid_physical_rmse", "full_grid_nll"):
        plot(fields, runs, metric, panels, args.output / f"field-{metric}.png")
    for field in ("uo", "vo", "thetao", "so"):
        panels = [
            (f"{field} {r['depth_m']:g} m", "channel", r["channel"])
            for r in channels[:77]
            if r["field"] == field
        ]
        for metric in ("full_grid_nmse", "full_grid_physical_rmse"):
            plot(channels, runs, metric, panels, args.output / f"{field}-{metric}.png")
    receipt = {
        "runs": [
            {
                "path": r["path"],
                "max_update": max(v["validation"]["step"] for v in r["vals"]),
                "complete": r["complete"],
                "provenance": r["provenance"],
            }
            for r in runs
        ],
        "sampling_signature": runs[0]["signature"],
        "physical_rmse_definition": "sqrt(mean_depth(std_channel^2 * full_grid_nmse)); land retained in denominator",
        "small_classifier": "Comparison restricted to full-state artifacts because its validation samples differ.",
    }
    with gzip.open(args.output / "curve-receipt.json.gz", "wt") as stream:
        stream.write(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
