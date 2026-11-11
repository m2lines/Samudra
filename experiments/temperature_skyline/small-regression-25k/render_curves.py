# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""CPU curves for the matched small regression and completed classifiers.

Requires production provenance even for partial runs. Qualification data are not
production results. Regression has no categorical probabilities or NLL.
"""

import argparse
import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "soft_prior_curve_helpers", HERE.parent / "soft-priors-25k/render_curves.py"
)
assert SPEC is not None and SPEC.loader is not None
helpers = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helpers)
plt = helpers.plt
REG = "all-fields-regression-25k"
DEFAULT_CE = Path(
    "/home/jder/.local/state/codex-async-followup/workflows/skyline-soft-priors-25k-artifacts/results"
)


def load_runs(regression, classifiers):
    runs = [helpers.read_run(classifiers / name, name) for name in helpers.ARMS]
    helpers.check_runs(runs)
    for run in runs:
        cfg = run["provenance"]["config"]
        if (
            cfg["model"]["objective"] != "absolute_ce"
            or cfg["model"]["bins"] != 8192
            or cfg["model"]["head_init_sigma"] != 4.0
            or cfg["categorical_target"] != "two_hot"
        ):
            raise ValueError(
                "Expected the completed 8192-bin two-hot/smooth-initialization classifiers"
            )
    # The generic reader permits missing categorical metrics for benchmark input;
    # this regression is subsequently checked as the matched all-fields control.
    reg = helpers.read_run(regression, REG, benchmark=True)
    reg.update(
        benchmark=False,
        matched_validation_samples=True,
        likelihood_comparable=False,
        label="All-fields regression",
    )
    ref = runs[1]
    cfg, ce = reg["provenance"]["config"], ref["provenance"]["config"]
    if cfg["qualification"] or cfg["max_updates"] != 25000:
        raise ValueError(
            "Regression input must have fresh 25k production configuration, not qualification"
        )
    if (
        cfg["model"]["objective"] != "regression"
        or cfg["temperature_loss_only"]
        or cfg["arm"] != "all_fields"
    ):
        raise ValueError("Expected supervised all-fields scalar regression")
    if cfg["model"]["head_init_sigma"] != 0 or cfg["discrete_readout_only"]:
        raise ValueError("Regression must have an ordinary scalar head")
    allowed = {
        "model",
        "categorical_target",
        "discrete_readout_only",
        "max_allocated_seconds",
    }
    helpers.same(
        {k: v for k, v in cfg.items() if k not in allowed},
        {k: v for k, v in ce.items() if k not in allowed},
        "production config outside documented objective/head/cap changes",
    )
    model_allowed = {"objective", "head_init_sigma"}
    helpers.same(
        {k: v for k, v in cfg["model"].items() if k not in model_allowed},
        {k: v for k, v in ce["model"].items() if k not in model_allowed},
        "backbone configuration",
    )
    helpers.same(
        reg["signature"],
        ref["signature"],
        "all 77 validation channels/origins/counts/masks/persistence",
    )
    for key in ("mean", "std", "metadata_sha256"):
        helpers.same(
            reg["provenance"]["source"][key],
            ref["provenance"]["source"][key],
            "normalization/source " + key,
        )
    helpers.same(
        reg["provenance"]["source_commit"],
        ref["provenance"]["source_commit"],
        "training source commit",
    )
    for v in reg["vals"]:
        helpers.same(
            v["validation"]["metric"],
            ref["vals"][0]["validation"]["metric"],
            "metric definition",
        )
        if len(v["validation"]["per_channel"]) != 77:
            raise ValueError("Regression needs all 77 channels")
        if any("nll" in key for row in v["validation"]["per_channel"] for key in row):
            raise ValueError("Regression receipts must not claim categorical NLL")
    runs[0]["label"] = "T-only classifier"
    runs[1]["label"] = "All-fields classifier"
    return runs + [reg]


def make_tables(runs):
    channels, fields, training = helpers.tables(runs)
    objectives = {
        r["name"]: r["provenance"]["config"]["model"]["objective"] for r in runs
    }
    for row in channels + fields + training:
        reg = objectives[row["run"]] == "regression"
        row.update(
            objective="normalized_MSE" if reg else "two_hot_cross_entropy",
            readout="continuous_scalar" if reg else "argmax_fixed_center",
            categorical_probabilities_available=not reg,
        )
    aggregate = []
    for run in runs:
        for record in run["vals"]:
            rows = [
                r
                for r in channels
                if r["run"] == run["name"]
                and r["step"] == record["validation"]["step"]
                and r["weights"] == record["weights"]
            ]
            base = {
                k: v
                for k, v in rows[0].items()
                if k
                in {
                    "run",
                    "label",
                    "step",
                    "weights",
                    "objective",
                    "readout",
                    "categorical_probabilities_available",
                    "likelihood_comparable",
                    "matched_validation_samples",
                }
            }
            mse = float(np.mean([r["full_grid_nmse"] for r in rows]))
            persistence = float(
                np.mean([r["persistence_full_grid_nmse"] for r in rows])
            )
            aggregate.append(
                base
                | dict(
                    channels=77,
                    supervised=all(r["supervised"] for r in rows),
                    full_grid_nmse=mse,
                    normalized_rmse=float(np.sqrt(mse)),
                    persistence_full_grid_nmse=persistence,
                    persistence_normalized_rmse=float(np.sqrt(persistence)),
                )
            )
    return channels, fields, training, aggregate


def finish(fig, axes, runs, output, filename, title):
    handles, labels = [], []
    for ax in np.asarray(axes).flat:
        h, names = ax.get_legend_handles_labels()
        for handle, label in zip(h, names, strict=True):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        fontsize=8,
        frameon=False,
    )
    budgets = "; ".join(
        f"{r['label']}: max recorded {max(x['validation']['step'] for x in r['vals']):,}"
        for r in runs
    )
    fig.suptitle(title + "\n" + budgets, fontsize=10)
    fig.text(
        0.5,
        0.005,
        "Step 0 retained in CSV; log axis starts at 1. Solid EMA, dashed raw. OM4/full-grid metrics. Regression has no NLL.",
        ha="center",
        fontsize=7,
    )
    fig.tight_layout(rect=(0, 0.085, 1, 0.94))
    fig.savefig(output / filename, dpi=135)
    plt.close(fig)


def render(runs, channels, fields, aggregate, output):
    temp = [r for r in fields if r["field"] == "thetao"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, rows, metric, title, label in (
        (
            axes[0],
            temp,
            "normalized_rmse",
            "Temperature: 19 channels",
            "sqrt(mean normalized MSE)",
        ),
        (
            axes[1],
            aggregate,
            "normalized_rmse",
            "All fields: 77 supervised channels",
            "sqrt(mean normalized MSE)",
        ),
        (
            axes[2],
            temp,
            "full_grid_nll",
            "Temperature classifier likelihood",
            "Hard NLL (nats/full-grid cell)",
        ),
    ):
        helpers.plot_panel(ax, rows, runs, metric, title, label)
    finish(
        fig,
        axes,
        runs,
        output,
        "aggregate-curves.png",
        "Matched evaluation; fixed-budget final EMA is primary",
    )
    for field in helpers.FIELDS[:-1]:
        fig, axes = plt.subplots(4, 5, figsize=(15, 11))
        for depth, ax in enumerate(axes.flat):
            if depth == 19:
                ax.set_axis_off()
                continue
            rows = [r for r in channels if r["channel"] == f"{field}_{depth}"]
            helpers.plot_panel(
                ax,
                rows,
                runs,
                "normalized_rmse",
                f"{rows[0]['depth_m']:g} m",
                "Normalized full-grid RMSE",
            )
        finish(
            fig,
            axes,
            runs,
            output,
            f"{field}-depth-rmse.png",
            f"{field}: all 19 depths; supervised predictions only",
        )
    fig, axes = plt.subplots(2, 5, figsize=(16, 8))
    for j, field in enumerate(helpers.FIELDS):
        rows = [r for r in fields if r["field"] == field]
        helpers.plot_panel(
            axes[0, j],
            rows,
            runs,
            "normalized_rmse",
            field,
            "Normalized full-grid RMSE",
        )
        helpers.plot_panel(
            axes[1, j],
            rows,
            runs,
            "full_grid_nll",
            field,
            "Classifier NLL (nats/full-grid cell)",
        )
    finish(
        fig,
        axes,
        runs,
        output,
        "field-curves.png",
        "Field aggregates; T-only auxiliary outputs excluded",
    )
    fig, axes = plt.subplots(4, 5, figsize=(15, 11))
    for depth, ax in enumerate(axes.flat):
        if depth == 19:
            ax.set_axis_off()
            continue
        rows = [r for r in channels if r["channel"] == f"thetao_{depth}"]
        helpers.plot_panel(
            ax,
            rows,
            runs,
            "full_grid_nll",
            f"{rows[0]['depth_m']:g} m",
            "Classifier hard NLL",
        )
    finish(
        fig,
        axes,
        runs,
        output,
        "temperature-depth-nll.png",
        "Classifier likelihood by depth; regression has no probability output",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regression",
        type=Path,
        required=True,
        help="all-fields-regression-25k artifact directory",
    )
    parser.add_argument("--classifiers", type=Path, default=DEFAULT_CE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = load_runs(args.regression, args.classifiers)
    channels, fields, training, aggregate = make_tables(runs)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("validation-channels.csv.gz", channels),
        ("validation-fields.csv", fields),
        ("validation-aggregate.csv", aggregate),
        ("training-progress.csv", training),
    ):
        helpers.write_csv(args.output / name, rows)
    render(runs, channels, fields, aggregate, args.output)
    receipt = helpers.summary(runs)
    receipt["notes"] = [
        "Partial curves are allowed but qualification input is rejected. Fixed final 25000-update EMA is primary; max observed update alone is not completion.",
        "Exact matched validation origins/counts/persistence and source normalization are required. Backbone configuration matches; regression changes scalar head, parameter count, objective and head initialization.",
        "Normalized RMSE is sqrt(mean normalized-channel MSE); it is not degrees C, PSU or m/s. Full-grid denominators and equal channel weights are retained.",
        "Regression uses MSE and a continuous scalar readout. Classifiers use two-hot cross entropy and fixed-center argmax. NLL is absent for regression; classifier likelihood does not establish calibration.",
        "All 77 raw/EMA channels, including T-only unsupervised auxiliaries, and update zero are retained in CSV; unsupervised channels do not enter scientific plot comparisons.",
        "Logged elapsed time is not allocated GPU cost; use separate accounting including qualification and failures.",
    ]
    (args.output / "curve-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    with gzip.open(args.output / "curve-provenance.json.gz", "wt") as f:
        json.dump({r["name"]: r["provenance"] for r in runs}, f)
    for file in args.output.iterdir():
        if file.suffix in {".png", ".csv", ".gz", ".json"}:
            file.with_name(file.name + ".license").write_text(
                "SPDX-FileCopyrightText: 2026 Samudra Authors\n\nSPDX-License-Identifier: CC-BY-4.0\n"
            )


if __name__ == "__main__":
    main()
