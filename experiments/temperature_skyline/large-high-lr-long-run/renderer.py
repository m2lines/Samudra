# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""CPU validation report for the fresh large 60k high-LR classifier run."""

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
    "skyline-large-classifier-artifacts/all-fields-25k"
)
FIELDS = ("uo", "vo", "thetao", "so", "zos")
ENDPOINTS = (25000, 50000, 55000, 60000)


def load_runs(root, historical, allow_partial=False):
    run = h.read_run(root / "all-fields-60k", "all-fields-60k")
    reference = h.read_run(historical, "historical-25k")
    config = run["provenance"]["config"]
    expected = json.loads((HERE / "train.json").read_text())
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
    for key in inherited + (
        "learning_rate_schedule",
        "learning_rate",
        "schedule_updates",
        "max_updates",
        "cooldown_start",
        "final_learning_rate",
        "qualification",
    ):
        h.same(config[key], expected[key], "60k scientific config " + key)
    if config.get("branch_checkpoint") or config.get("continuation_parent"):
        raise ValueError("Expected a fresh run")
    for key in inherited:
        h.same(
            config[key],
            reference["provenance"]["config"][key],
            "historical recipe " + key,
        )
    h.same(
        run["signature"],
        reference["signature"],
        "validation dates/counts/masks/persistence",
    )
    for key in ("mean", "std", "metadata_sha256"):
        h.same(
            run["provenance"]["source"][key],
            reference["provenance"]["source"][key],
            "source " + key,
        )
    for item in (run, reference):
        for record in item["vals"]:
            h.same(
                record["validation"]["metric"],
                run["vals"][0]["validation"]["metric"],
                "metric definition",
            )
    for weights in ("raw", "ema"):
        steps = {
            v["validation"]["step"] for v in run["vals"] if v["weights"] == weights
        }
        if 0 not in steps or any(step < 0 or step > 60000 for step in steps):
            raise ValueError("Expected initial validation and updates within 0..60000")
        if not allow_partial and 60000 not in steps:
            raise ValueError(
                "Final raw/EMA 60k validation is pending; use --allow-partial for a progress report"
            )
    reference["vals"] = [
        v
        for v in reference["vals"]
        if v["weights"] == "ema" and v["validation"]["step"] == 25000
    ]
    if len(reference["vals"]) != 1:
        raise ValueError("Historical final 25k EMA validation is required")
    run.update(
        label="Fresh high LR",
        matched_validation_samples=True,
        likelihood_comparable=True,
    )
    reference.update(
        label="Historical 25k EMA",
        benchmark=True,
        matched_validation_samples=True,
        likelihood_comparable=True,
    )
    return [run, reference]


def make_tables(runs):
    channels, fields, training = large.tables(runs)
    summary = []
    for aggregation, rows in (("channel", channels), ("field", fields)):
        for row in rows:
            row["is_historical_benchmark"] = row["run"] == "historical-25k"
            row["readout"] = "argmax_fixed_center"
            if row["step"] in ENDPOINTS:
                summary.append(
                    row | {"aggregation": aggregation, "endpoint_update": row["step"]}
                )
            if row["is_historical_benchmark"]:
                summary.append(
                    {
                        "run": "persistence",
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


def plot(rows, panels, metric, output, status, cooldown=False):
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
        for weights, style in (("raw", "--"), ("ema", "-")):
            series = sorted(
                (
                    r
                    for r in selected
                    if r["run"] == "all-fields-60k"
                    and r["weights"] == weights
                    and r["step"] >= (50000 if cooldown else 1)
                ),
                key=lambda r: r["step"],
            )
            if series:
                ax.plot(
                    [r["step"] for r in series],
                    [r[metric] for r in series],
                    color="#0072b2",
                    ls=style,
                    marker=".",
                    lw=1.1,
                    label=f"Fresh high LR {weights}",
                )
        ref = next(r for r in selected if r["run"] == "historical-25k")
        ax.axhline(ref[metric], color="#777777", label="Historical 25k EMA")
        if metric == "full_grid_nmse":
            ax.axhline(
                ref["persistence_full_grid_nmse"],
                color="black",
                ls=":",
                label="Persistence",
            )
            ax.set_yscale("log")
        if cooldown and not any(
            r["run"] == "all-fields-60k" and r["step"] >= 50000 for r in selected
        ):
            ax.text(
                0.5,
                0.85,
                "Cooldown validation pending",
                transform=ax.transAxes,
                ha="center",
                fontsize=8,
            )
        ax.axvspan(50000, 60000, color="#c4d7ee", alpha=0.3, label="Cooldown")
        ax.set(
            xscale="linear" if cooldown else "log",
            xlim=(50000 if cooldown else 1, 60000),
            xlabel="Optimizer update",
            ylabel=metric,
            title=title,
        )
        ax.grid(alpha=0.2)
    for ax in list(axes.flat)[len(panels) :]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    fig.suptitle(
        f"Large classifier · {status} · raw dashed / EMA solid\nHistorical 25k reference has a different LR schedule and training budget"
    )
    fig.tight_layout(
        rect=(0, 0.08 if nrows > 1 else 0.18, 1, 0.94 if nrows > 1 else 0.88)
    )
    fig.savefig(output, dpi=120)
    h.plt.close(fig)


def learning_rates(runs, output):
    fig, axes = h.plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    rows: list[dict[str, str | int | float]] = []
    for run, color, end in zip(
        runs, ("#0072b2", "#777777"), (60000, 25000), strict=True
    ):
        steps = np.arange(1, end + 1)
        rates = [
            scheduled_learning_rate(run["provenance"]["config"], int(step))
            for step in steps
        ]
        rows.extend(
            dict(run=run["name"], step=int(step), learning_rate=rate)
            for step, rate in zip(steps, rates, strict=True)
        )
        for ax in axes:
            ax.plot(steps, rates, color=color, label=run["label"])
    for ax in axes:
        ax.axvspan(50000, 60000, color="#c4d7ee", alpha=0.3)
        ax.set(xlabel="Optimizer update", ylabel="Configured learning rate")
        ax.grid(alpha=0.2)
    axes[0].set(xscale="log", xlim=(1, 60000), title="Configured schedules")
    axes[1].set(xlim=(49000, 60000), title="Cooldown to 2e-5")
    axes[0].legend(fontsize=8)
    fig.savefig(output / "learning-rate.png", dpi=120)
    h.plt.close(fig)
    h.write_csv(output / "learning-rate.csv.gz", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Artifact directory containing all-fields-60k",
    )
    parser.add_argument("--historical", type=Path, default=HISTORICAL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    runs = load_runs(args.root, args.historical, args.allow_partial)
    channels, fields, training, summary = make_tables(runs)
    missing = [
        {"weights": weights, "step": step}
        for weights in ("raw", "ema")
        for step in ENDPOINTS
        if not any(
            v["weights"] == weights and v["validation"]["step"] == step
            for v in runs[0]["vals"]
        )
    ]
    observed = max(v["validation"]["step"] for v in runs[0]["vals"])
    status = (
        f"partial through {observed:,}"
        if any(m["step"] == 60000 for m in missing)
        else "final 60k validation"
    )
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
        plot(fields, panels, metric, args.output / f"fields-{metric}.png", status)
        plot(
            fields,
            panels,
            metric,
            args.output / f"cooldown-fields-{metric}.png",
            status,
            cooldown=True,
        )
        for field in FIELDS:
            initial = [
                r
                for r in channels
                if r["run"] == "all-fields-60k"
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
            plot(
                channels, panels, metric, args.output / f"{field}-{metric}.png", status
            )
    learning_rates(runs, args.output)
    receipt = dict(
        status=status,
        missing_fixed_endpoints=missing,
        validation_signature=runs[0]["signature"],
        runs=[
            dict(
                name=r["name"],
                path=r["path"],
                complete=r["complete"],
                provenance=r["provenance"],
            )
            for r in runs
        ],
        metric_notes="Full-grid NMSE retains land denominator. Physical RMSE is sqrt(mean_depth(std_channel^2*NMSE)). Hard NLL is taken directly from validation, independently of two-hot training loss. Step zero remains in CSV. Missing endpoints are omitted.",
        interpretation="Single-seed experiment. Historical 25k EMA has a different LR schedule and training budget. NLL curves do not establish calibration.",
    )
    with gzip.open(args.output / "render-receipt.json.gz", "wt") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
