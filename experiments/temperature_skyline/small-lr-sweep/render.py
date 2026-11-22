# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""CPU validation curves for fresh small-classifier learning-rate experiments."""

import argparse
import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np

from samudra.skyline.branching import scheduled_learning_rate

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "large_curve_helpers", HERE.parent / "large-all-fields-classifier/render_curves.py"
)
assert SPEC is not None and SPEC.loader is not None
large = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(large)
h = large.h
HISTORICAL = Path(
    "/home/jder/.local/state/codex-async-followup/workflows/"
    "skyline-soft-priors-25k-artifacts/results/all-fields-both-25k"
)
ARMS = {"1x-train": 2e-4, "2x-train": 4e-4, "4x-train": 8e-4}
FIELDS = ("uo", "vo", "thetao", "so", "zos")
COLORS = ("#0072b2", "#d55e00", "#009e73", "#777777")


def load_runs(root, historical):
    reference = h.read_run(historical, "historical-25k")
    reference.update(
        label="Historical cosine",
        benchmark=True,
        matched_validation_samples=True,
        likelihood_comparable=True,
    )
    runs = []
    inherited = (
        "model",
        "arm",
        "seed",
        "validation_seed",
        "crop",
        "microbatch",
        "physical_microbatch",
        "prefetch_microbatches",
        "accumulation",
        "world_size",
        "samples_per_frame",
        "coverage",
        "ema",
        "weight_decay",
        "warmup",
        "validation_dates",
        "validation_points_per_depth",
        "vocabulary_bounds",
        "categorical_target",
        "discrete_readout_only",
    )
    for name, rate in ARMS.items():
        run = h.read_run(root / name, name)
        cfg = run["provenance"]["config"]
        with gzip.open(HERE / f"{name}.json.gz", "rt") as handle:
            expected = json.load(handle)
        for key in inherited + (
            "learning_rate_schedule",
            "learning_rate",
            "schedule_updates",
            "max_updates",
            "cooldown_start",
            "final_learning_rate",
            "qualification",
        ):
            h.same(cfg[key], expected[key], f"{name} {key}")
        h.same(cfg["learning_rate"], rate, name + " peak LR")
        if cfg.get("branch_checkpoint") or cfg.get("continuation_parent"):
            raise ValueError("Fresh sweep runs must start from initialization")
        for key in inherited:
            h.same(
                cfg[key],
                reference["provenance"]["config"][key],
                f"{name} historical recipe {key}",
            )
        for weights in ("raw", "ema"):
            if not any(
                v["weights"] == weights and v["validation"]["step"] == 0
                for v in run["vals"]
            ):
                raise ValueError(f"{name} missing step-zero {weights} validation")
        run.update(
            label=f"{name.split('-')[0]} peak {rate:g}",
            matched_validation_samples=True,
            likelihood_comparable=True,
        )
        runs.append(run)
    for run in runs + [reference]:
        h.same(
            run["signature"],
            reference["signature"],
            run["name"] + " samples/persistence",
        )
        for key in ("mean", "std", "metadata_sha256"):
            h.same(
                run["provenance"]["source"][key],
                reference["provenance"]["source"][key],
                run["name"] + " source " + key,
            )
        for record in run["vals"]:
            value = record["validation"]
            if not 0 <= value["step"] <= (25000 if run is reference else 30000):
                raise ValueError("Validation update outside configured run")
            h.same(
                value["metric"],
                reference["vals"][0]["validation"]["metric"],
                "metric definition",
            )
    for run in runs[1:]:
        h.same(
            run["provenance"]["source_commit"],
            runs[0]["provenance"]["source_commit"],
            "fresh-arm source commit",
        )
    if not all(
        any(
            v["weights"] == kind and v["validation"]["step"] == 25000
            for v in reference["vals"]
        )
        for kind in ("raw", "ema")
    ):
        raise ValueError("Historical run needs both final 25k weight evaluations")
    return runs + [reference]


def make_tables(runs):
    channels, fields, training = large.tables(runs)
    for row in channels + fields:
        row["is_historical_benchmark"] = row["run"] == "historical-25k"
        row["phase"] = (
            "historical_cosine"
            if row["is_historical_benchmark"]
            else ("cooldown" if row["step"] > 25000 else "warmup_or_stable")
        )
        row["readout"] = "argmax_fixed_center"
    summary = []
    for aggregation, rows in (("channel", channels), ("field", fields)):
        for row in rows:
            if row["step"] in (25000, 30000):
                summary.append(
                    row | {"aggregation": aggregation, "endpoint_update": row["step"]}
                )
        for row in rows:
            if (
                row["run"] == "historical-25k"
                and row["weights"] == "ema"
                and row["step"] == 25000
            ):
                summary.append(
                    {
                        "run": "persistence",
                        "label": "Matched persistence",
                        "weights": "persistence",
                        "field": row["field"],
                        "channel": row.get("channel", ""),
                        "depth_m": row.get("depth_m", ""),
                        "aggregation": aggregation,
                        "physical_units": row["physical_units"],
                        "full_grid_nmse": row["persistence_full_grid_nmse"],
                        "full_grid_physical_rmse": row[
                            "persistence_full_grid_physical_rmse"
                        ],
                    }
                )
    return channels, fields, training, summary


def plot(rows, panels, runs, metric, output, cooldown=False):
    ncols = min(4, len(panels))
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = h.plt.subplots(
        nrows,
        ncols,
        figsize=(10, 5.2) if ncols == 1 else (18, 3.3 * nrows),
        squeeze=False,
    )
    for ax, (title, key, value) in zip(axes.flat, panels, strict=False):
        selected = [r for r in rows if r[key] == value]
        for index, run in enumerate(runs):
            for weights, style in (("raw", "--"), ("ema", "-")):
                series = sorted(
                    (
                        r
                        for r in selected
                        if r["run"] == run["name"]
                        and r["weights"] == weights
                        and r["step"] >= (25000 if cooldown else 1)
                    ),
                    key=lambda r: r["step"],
                )
                if cooldown and run["name"] == "historical-25k":
                    if series:
                        ax.axhline(
                            series[-1][metric],
                            color=COLORS[index],
                            ls=style,
                            label=f"Historical 25k {weights}",
                        )
                else:
                    ax.plot(
                        [r["step"] for r in series],
                        [r[metric] for r in series],
                        color=COLORS[index],
                        ls=style,
                        marker=".",
                        lw=1.1,
                        label=f"{run['label']} {weights}",
                    )
        if metric == "full_grid_nmse":
            ax.axhline(
                selected[0]["persistence_full_grid_nmse"],
                color="black",
                ls=":",
                label="Persistence",
            )
            ax.set_yscale("log")
        ax.axvspan(25000, 30000, color="#c4d7ee", alpha=0.3, label="Cooldown")
        ax.set(
            xscale="linear" if cooldown else "log",
            xlim=(25000 if cooldown else 1, 30000),
            xlabel="Optimizer update",
            ylabel=metric,
            title=title,
        )
        ax.grid(alpha=0.2)
    for ax in list(axes.flat)[len(panels) :]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7)
    fig.suptitle(
        "Fresh small-classifier LR sweep · raw dashed / EMA solid · matched validation · single seed"
    )
    fig.tight_layout(
        rect=(0, 0.10 if nrows > 1 else 0.24, 1, 0.96 if nrows > 1 else 0.92)
    )
    fig.savefig(output, dpi=120)
    h.plt.close(fig)


def learning_rates(runs, output):
    fig, axes = h.plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    rows: list[dict[str, str | int | float]] = []
    for index, run in enumerate(runs):
        cfg = run["provenance"]["config"]
        end = 25000 if run["name"] == "historical-25k" else 30000
        steps = np.arange(1, end + 1)
        rates = [scheduled_learning_rate(cfg, int(step)) for step in steps]
        rows.extend(
            dict(run=run["name"], step=int(step), learning_rate=rate)
            for step, rate in zip(steps, rates, strict=True)
        )
        for ax in axes:
            ax.plot(steps, rates, color=COLORS[index], label=run["label"])
    for ax in axes:
        ax.axvspan(25000, 30000, color="#c4d7ee", alpha=0.3)
        ax.set(xlabel="Optimizer update", ylabel="Configured learning rate")
        ax.grid(alpha=0.2)
    axes[0].set(xscale="log", xlim=(1, 30000), title="Warmup and full schedule")
    axes[1].set(xlim=(24000, 30000), title="Cooldown to common 2e-5 floor")
    axes[0].legend(fontsize=8)
    fig.savefig(output / "learning-rate.png", dpi=120)
    h.plt.close(fig)
    h.write_csv(output / "learning-rate.csv.gz", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--historical", type=Path, default=HISTORICAL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = load_runs(args.root, args.historical)
    channels, fields, training, summary = make_tables(runs)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("validation-channels.csv.gz", channels),
        ("validation-fields.csv.gz", fields),
        ("fixed-endpoints.csv.gz", summary),
        ("training-progress.csv.gz", training),
    ):
        h.write_csv(args.output / name, rows)
    for metric in ("full_grid_nmse", "full_grid_nll"):
        panels = [(f, "field", f) for f in FIELDS]
        plot(fields, panels, runs, metric, args.output / f"fields-{metric}.png")
        plot(
            fields,
            panels,
            runs,
            metric,
            args.output / f"cooldown-fields-{metric}.png",
            cooldown=True,
        )
        for field in FIELDS:
            initial = [
                r
                for r in channels
                if r["run"] == runs[0]["name"]
                and r["weights"] == "ema"
                and r["step"] == 0
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
                for r in initial
            ]
            plot(channels, panels, runs, metric, args.output / f"{field}-{metric}.png")
    learning_rates(runs, args.output)
    receipt = {
        "runs": [
            {
                "name": r["name"],
                "path": r["path"],
                "max_update": max(v["validation"]["step"] for v in r["vals"]),
                "complete": r["complete"],
                "provenance": r["provenance"],
            }
            for r in runs
        ],
        "missing_fixed_endpoints": [
            {"run": r["name"], "weights": weights, "step": step}
            for r in runs[:3]
            for weights in ("raw", "ema")
            for step in (25000, 30000)
            if not any(
                v["weights"] == weights and v["validation"]["step"] == step
                for v in r["vals"]
            )
        ],
        "validation_signature": runs[0]["signature"],
        "metric_notes": "Full-grid NMSE retains land in denominator. Physical RMSE is sqrt(mean_depth(std_channel^2*NMSE)). Hard categorical NLL comes directly from validation, independently of the two-hot training objective. Step zero is retained in CSV. Missing fixed endpoints are omitted, without last-value substitution.",
        "historical_benchmark": "Historical full-training cosine schedule ends at 25k. Its Torch smoothed-head initialization differs from the current NumPy implementation within floating-point roundoff. Matched samples do not establish identical initialization.",
        "interpretation": "Experimental single-seed comparison. These NLL curves do not establish probability calibration.",
    }
    with gzip.open(args.output / "render-receipt.json.gz", "wt") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
