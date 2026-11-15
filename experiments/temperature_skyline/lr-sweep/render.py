# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""CPU curves for constant-LR branches from the large classifier's 25k checkpoint.

Accepts partial results and reports missing fixed 3,000-update endpoints explicitly.
Hard categorical NLL is preserved independently of the two-hot training loss.
"""

import argparse
import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "large_curve_helpers", HERE.parent / "large-all-fields-classifier/render_curves.py"
)
assert SPEC is not None and SPEC.loader is not None
large = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(large)
h = large.h
CLASSIFIER = Path(
    "/home/jder/.local/state/codex-async-followup/workflows/"
    "skyline-large-classifier-artifacts/all-fields-25k"
)
ARMS = {"1x-train": 2e-4, "2x-train": 4e-4, "4x-train": 8e-4}
FIELDS = ("uo", "vo", "thetao", "so", "zos")
STEPS = {25000, 25050, 25100, 25250, 25500, 26000, 27000, 28000}


def load_runs(root, classifier, regression):
    references = large.load_runs(classifier, regression)
    reference = references[0]
    for run in references:
        run["vals"] = [v for v in run["vals"] if v["validation"]["step"] == 25000]
        if {v["weights"] for v in run["vals"]} != {"raw", "ema"}:
            raise ValueError("Both reference weights must have final 25k validation")
    branches = []
    for name, rate in ARMS.items():
        run = h.read_run(root / name, name)
        cfg = run["provenance"]["config"]
        with gzip.open(HERE / f"{name}.json.gz", "rt") as handle:
            expected = json.load(handle)
        for key in (
            "branch_checkpoint",
            "learning_rate_schedule",
            "learning_rate",
            "max_updates",
            "qualification",
        ):
            h.same(cfg[key], expected[key], f"{name} {key}")
        h.same(cfg["learning_rate"], rate, f"{name} constant LR")
        for key in (
            "model",
            "arm",
            "seed",
            "validation_seed",
            "crop",
            "microbatch",
            "physical_microbatch",
            "accumulation",
            "world_size",
            "samples_per_frame",
            "coverage",
            "ema",
            "weight_decay",
            "validation_dates",
            "validation_points_per_depth",
            "vocabulary_bounds",
            "categorical_target",
            "discrete_readout_only",
        ):
            h.same(
                cfg[key],
                reference["provenance"]["config"][key],
                f"{name} inherited {key}",
            )
        h.same(
            run["signature"],
            reference["signature"],
            f"{name} validation sampling/persistence",
        )
        for key in ("mean", "std", "metadata_sha256"):
            h.same(
                run["provenance"]["source"][key],
                reference["provenance"]["source"][key],
                f"{name} source {key}",
            )
        for record in run["vals"]:
            val = record["validation"]
            if val["step"] not in STEPS:
                raise ValueError(f"Unexpected validation update {val['step']}")
            h.same(
                val["metric"],
                reference["vals"][0]["validation"]["metric"],
                f"{name} metric",
            )
        for kind in ("raw", "ema"):
            if not any(
                v["weights"] == kind and v["validation"]["step"] == 25000
                for v in run["vals"]
            ):
                raise ValueError(f"{name} missing initial {kind} validation at 25k")
        run.update(
            label=f"{name.split('-')[0]} LR {rate:g}",
            matched_validation_samples=True,
            likelihood_comparable=True,
        )
        branches.append(run)
    return branches, references


def make_tables(branches, references):
    channels, fields, training = large.tables(branches + references)
    for row in channels + fields:
        row["additional_updates"] = row["step"] - 25000
        row["is_reference"] = row["run"] not in ARMS
        row["point_prediction_kind"] = (
            "continuous regression"
            if row["run"] == "original-regression"
            else "argmax-bin center"
        )
        if row["run"] in ARMS:
            row["constant_learning_rate"] = ARMS[row["run"]]
    summary = []
    for kind, rows in (("channel", channels), ("field", fields)):
        for row in rows:
            if row["is_reference"] or row["step"] == 28000:
                summary.append(
                    row
                    | {
                        "aggregation": kind,
                        "comparison_role": "reference_25000"
                        if row["is_reference"]
                        else "fixed_3000_additional",
                    }
                )
        for row in rows:
            if row["run"] == "large-classifier" and row["weights"] == "ema":
                summary.append(
                    {
                        "run": "persistence",
                        "label": "Matched persistence",
                        "weights": "persistence",
                        "field": row["field"],
                        "channel": row.get("channel", ""),
                        "depth_m": row.get("depth_m", ""),
                        "aggregation": kind,
                        "comparison_role": "persistence",
                        "physical_units": row["physical_units"],
                        "full_grid_nmse": row["persistence_full_grid_nmse"],
                        "full_grid_physical_rmse": row[
                            "persistence_full_grid_physical_rmse"
                        ],
                    }
                )
    return channels, fields, training, summary


def plot(rows, panels, branches, references, metric, output):
    ncols = min(4, len(panels))
    nrows = int(np.ceil(len(panels) / ncols))
    figsize = (10, 5.2) if ncols == 1 else (18, 3.3 * nrows)
    fig, axes = h.plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for ax, (label, key, value) in zip(axes.flat, panels, strict=False):
        selected = [r for r in rows if r[key] == value]
        for index, run in enumerate(branches):
            for weights, style in (("raw", "--"), ("ema", "-")):
                series = sorted(
                    (
                        r
                        for r in selected
                        if r["run"] == run["name"] and r["weights"] == weights
                    ),
                    key=lambda r: r["additional_updates"],
                )
                ax.plot(
                    [r["additional_updates"] for r in series],
                    [r[metric] for r in series],
                    color=("#0072b2", "#d55e00", "#009e73")[index],
                    ls=style,
                    marker=".",
                    lw=1.1,
                    label=f"{run['label']} {weights}",
                )
        for index, run in enumerate(references):
            for weights, style in (("raw", "--"), ("ema", "-")):
                ref = next(
                    r
                    for r in selected
                    if r["run"] == run["name"] and r["weights"] == weights
                )
                if metric in ref:
                    ax.axhline(
                        ref[metric],
                        color=("#777777", "#6a3d9a")[index],
                        ls=style,
                        lw=1,
                        label=f"25k {'classifier' if index == 0 else 'regressor'} {weights}",
                    )
        if metric == "full_grid_nmse":
            ax.axhline(
                selected[0]["persistence_full_grid_nmse"],
                color="black",
                ls=":",
                label="Persistence",
            )
            ax.set_yscale("log")
        ax.set(
            xlim=(0, 3000),
            xlabel="Additional optimizer updates after 25k",
            ylabel=metric,
            title=label,
        )
        ax.grid(alpha=0.2)
    for ax in list(axes.flat)[len(panels) :]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7)
    fig.suptitle(
        "Constant-LR branches · matched validation · raw dashed / EMA solid · single seed"
    )
    fig.tight_layout(
        rect=(0, 0.10 if nrows > 1 else 0.24, 1, 0.96 if nrows > 1 else 0.92)
    )
    fig.savefig(output, dpi=120)
    h.plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--classifier-reference", type=Path, default=CLASSIFIER)
    parser.add_argument("--regression-reference", type=Path, default=large.REFERENCE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    branches, references = load_runs(
        args.root, args.classifier_reference, args.regression_reference
    )
    channels, fields, training, summary = make_tables(branches, references)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("validation-channels.csv.gz", channels),
        ("validation-fields.csv.gz", fields),
        ("fixed-endpoint-summary.csv.gz", summary),
        ("training-progress.csv.gz", training),
    ):
        h.write_csv(args.output / name, rows)
    for metric in ("full_grid_nmse", "full_grid_nll"):
        plot(
            fields,
            [(f, "field", f) for f in FIELDS],
            branches,
            references,
            metric,
            args.output / f"fields-{metric}.png",
        )
        for field in FIELDS:
            first = [
                r
                for r in channels
                if r["run"] == branches[0]["name"]
                and r["step"] == 25000
                and r["weights"] == "ema"
                and r["field"] == field
            ]
            panels = [
                (
                    "SSH surface"
                    if field == "zos"
                    else f"{r['channel']} · {r['depth_m']:g} m",
                    "channel",
                    r["channel"],
                )
                for r in first
            ]
            plot(
                channels,
                panels,
                branches,
                references,
                metric,
                args.output / f"{field}-{metric}.png",
            )
    receipt = {
        "runs": [
            {
                "name": r["name"],
                "path": r["path"],
                "max_update": max(v["validation"]["step"] for v in r["vals"]),
                "complete": r["complete"],
                "provenance": r["provenance"],
            }
            for r in branches + references
        ],
        "missing_fixed_endpoints": [
            {"run": r["name"], "weights": kind}
            for r in branches
            for kind in ("raw", "ema")
            if not any(
                v["weights"] == kind and v["validation"]["step"] == 28000
                for v in r["vals"]
            )
        ],
        "validation_signature": branches[0]["signature"],
        "metric_notes": "MSE retains full-grid denominator. Physical RMSE is sqrt(mean_depth(std_channel^2*NMSE)). Hard categorical NLL is recorded directly from validation, independently of two-hot training loss. Regression and persistence have no categorical NLL.",
        "interpretation": "Experimental single-seed LR comparison. These NLL curves do not establish probability calibration. Fixed endpoint means 3000 additional updates; partial arms have no substituted endpoint.",
    }
    with gzip.open(args.output / "render-receipt.json.gz", "wt") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
