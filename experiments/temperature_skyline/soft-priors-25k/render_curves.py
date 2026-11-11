# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""CPU validation curves for the matched fresh soft-prior pair.

Accepts partial runs, labels their observed budget, and never calls inference.
Optional historical directories must use the same per-channel validation schema
and normalization. Their curves remain benchmarks, even with matched samples.
"""

import argparse
import csv
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARMS = ("temperature-both-25k", "all-fields-both-25k")
COLORS = ("#0072b2", "#d55e00", "#777777", "#009e73", "#cc79a7")
LABELS = {ARMS[0]: "New T only", ARMS[1]: "New all fields"}
FIELDS = ("thetao", "so", "uo", "vo", "zos")
METRICS = (
    "full_grid_nmse",
    "persistence_full_grid_nmse",
    "full_grid_nll",
    "wet_cell_nll",
)
NOTE = "Step 0 is retained in CSV; log update axis begins at 1. Solid: EMA; dashed: raw. OM4 targets; full-grid denominator."


def read_json(path):
    return json.loads(path.read_text())


def same(a, b, description):
    if a != b:
        raise ValueError(f"Mismatch: {description}")


def sample_signature(v):
    return dict(
        origins=v["origins"],
        channels=[
            {
                k: r[k]
                for k in (
                    "channel",
                    "depth",
                    "depth_m",
                    "count",
                    "wet_fraction",
                    "persistence_full_grid_nmse",
                )
            }
            for r in sorted(v["per_channel"], key=lambda x: x["channel"])
        ],
    )


def read_run(path, name, benchmark=False):
    provenance = read_json(path / "provenance-restart-0.json")
    vals = []
    for kind in ("raw", "ema"):
        files = sorted(path.glob(f"validation-{kind}-*.json"))
        if not files:
            raise ValueError(f"{path}: missing {kind} validation files")
        seen = set()
        for file in files:
            v = read_json(file)
            if not isinstance(v["step"], int) or v["step"] < 0 or v["step"] in seen:
                raise ValueError(f"{file}: invalid or duplicate update")
            seen.add(v["step"])
            if "per_channel" not in v:
                raise ValueError(
                    f"{file}: benchmark needs explicit per-channel normalized full-grid metrics; global-only summaries cannot be overlaid"
                )
            channels = [r["channel"] for r in v["per_channel"]]
            if len(set(channels)) != len(channels):
                raise ValueError(f"{file}: duplicate channels")
            expected = {f"thetao_{i}" for i in range(19)}
            if not expected.issubset(channels):
                raise ValueError(f"{file}: missing temperature depths")
            if not benchmark and set(channels) != (
                {f"{field}_{i}" for field in FIELDS[:-1] for i in range(19)} | {"zos"}
            ):
                raise ValueError(f"{file}: expected all 77 channels")
            for r in v["per_channel"]:
                for key in METRICS[:2]:
                    if key not in r or not np.isfinite(r[key]) or r[key] < 0:
                        raise ValueError(f"{file}/{r['channel']}: invalid {key}")
                for key in METRICS[2:]:
                    if key in r and (not np.isfinite(r[key]) or r[key] < 0):
                        raise ValueError(f"{file}/{r['channel']}: invalid {key}")
                if not benchmark and any(key not in r for key in METRICS):
                    raise ValueError(f"{file}: missing likelihood metric")
            vals.append(dict(weights=kind, validation=v))
    vals.sort(key=lambda x: (x["validation"]["step"], x["weights"]))
    signature = sample_signature(vals[0]["validation"])
    for v in vals:
        same(
            sample_signature(v["validation"]),
            signature,
            f"{name} validation sampling changed across updates/weights",
        )
    training_path = path / "training.jsonl"
    training = []
    if training_path.exists():
        training = [
            json.loads(line)
            for line in training_path.read_text().splitlines()
            if line.strip()
        ]
    complete_path = path / "COMPLETE.json"
    complete = read_json(complete_path) if complete_path.exists() else None
    return dict(
        name=name,
        path=str(path.resolve()),
        benchmark=benchmark,
        provenance=provenance,
        vals=vals,
        signature=signature,
        training=training,
        complete=complete,
    )


def check_runs(runs):
    a, b = runs[:2]
    same(
        a["signature"],
        b["signature"],
        "new arms validation origins/counts/masks/persistence",
    )
    for r in runs:
        same(
            r["vals"][0]["validation"]["metric"],
            a["vals"][0]["validation"]["metric"],
            f"{r['name']} metric definition",
        )
        for key in ("mean", "std", "metadata_sha256"):
            same(
                r["provenance"]["source"][key],
                a["provenance"]["source"][key],
                f"{r['name']} normalization/source {key}",
            )
    ca, cb = a["provenance"]["config"], b["provenance"]["config"]
    if not ca["temperature_loss_only"] or cb["temperature_loss_only"]:
        raise ValueError(
            "New arms must be temperature-only and all-fields respectively"
        )
    for key in (
        "model",
        "seed",
        "crop",
        "microbatch",
        "accumulation",
        "physical_microbatch",
        "coverage",
        "learning_rate",
        "warmup",
        "schedule_updates",
        "categorical_target",
        "vocabulary_bounds",
        "prefetch_microbatches",
        "cache_validation_frames",
    ):
        same(ca[key], cb[key], f"new arms config {key}")
    for r in runs:
        cfg = r["provenance"]["config"]
        r["matched_validation_samples"] = r["signature"] == a["signature"]
        r["likelihood_comparable"] = (
            cfg.get("model", {}).get("objective") == ca["model"]["objective"]
            and cfg.get("model", {}).get("bins") == ca["model"]["bins"]
            and cfg.get("vocabulary_bounds") == ca.get("vocabulary_bounds")
        )
        r["label"] = LABELS.get(
            r["name"],
            r["name"]
            + " (historical; "
            + (
                "same samples"
                if r["matched_validation_samples"]
                else "unmatched samples"
            )
            + ")",
        )


def tables(runs):
    channels, fields, training = [], [], []
    for run in runs:
        cfg = run["provenance"]["config"]
        t_only = (
            cfg.get("temperature_loss_only", False) or cfg.get("arm") == "temperature"
        )
        for record in run["vals"]:
            v = record["validation"]
            base = dict(
                run=run["name"],
                label=run["label"],
                benchmark=run["benchmark"],
                matched_validation_samples=run["matched_validation_samples"],
                likelihood_comparable=run["likelihood_comparable"],
                weights=record["weights"],
                step=v["step"],
                validation_origins=";".join(map(str, v["origins"])),
            )
            for row in v["per_channel"]:
                field = row["channel"].split("_")[0]
                channels.append(
                    base
                    | dict(field=field, supervised=not t_only or field == "thetao")
                    | row
                    | dict(
                        normalized_rmse=float(np.sqrt(row["full_grid_nmse"])),
                        persistence_normalized_rmse=float(
                            np.sqrt(row["persistence_full_grid_nmse"])
                        ),
                    )
                )
            for field in FIELDS:
                rows = [
                    r for r in v["per_channel"] if r["channel"].split("_")[0] == field
                ]
                if not rows:
                    continue
                metrics = {
                    k: float(np.mean([r[k] for r in rows]))
                    for k in METRICS
                    if all(k in r for r in rows)
                }
                fields.append(
                    base
                    | dict(
                        field=field,
                        supervised=not t_only or field == "thetao",
                        channels=len(rows),
                    )
                    | metrics
                    | dict(
                        normalized_rmse=float(np.sqrt(metrics["full_grid_nmse"])),
                        persistence_normalized_rmse=float(
                            np.sqrt(metrics["persistence_full_grid_nmse"])
                        ),
                    )
                )
        for row in run["training"]:
            training.append(
                dict(run=run["name"])
                | {
                    k: v
                    for k, v in row.items()
                    if v is None or isinstance(v, (str, int, float, bool))
                }
            )
    return channels, fields, training


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_panel(ax, rows, runs, metric, title, ylabel):
    for i, run in enumerate(runs):
        if "nll" in metric and not run["likelihood_comparable"]:
            continue
        for kind, style in (("ema", "-"), ("raw", "--")):
            vals = sorted(
                (
                    r
                    for r in rows
                    if r["run"] == run["name"]
                    and r["weights"] == kind
                    and r["step"] >= 1
                    and r["supervised"]
                    and metric in r
                ),
                key=lambda x: x["step"],
            )
            if not vals:
                continue
            ax.plot(
                [r["step"] for r in vals],
                [r[metric] for r in vals],
                color=COLORS[i % len(COLORS)],
                ls=style,
                marker=".",
                markersize=3,
                lw=1.1,
                alpha=0.9 if kind == "ema" else 0.6,
                label=run["label"] if kind == "ema" else "_nolegend_",
            )
    if metric in ("normalized_rmse", "full_grid_nmse"):
        baseline = next(r for r in rows if r["run"] == ARMS[0])
        ax.axhline(
            baseline["persistence_" + metric],
            color="black",
            ls=":",
            lw=1,
            label="Matched persistence",
        )
    ax.set_xscale("log")
    ax.set_xlim(left=1)
    if metric in ("normalized_rmse", "full_grid_nmse"):
        ax.set_yscale("log")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Optimizer update", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.2)


def finish(fig, axes, runs, output, name, title):
    handles, labels = [], []
    for ax in np.asarray(axes).flat:
        h, labs = ax.get_legend_handles_labels()
        for handle, label in zip(h, labs, strict=True):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=2,
        fontsize=8,
        frameon=False,
    )
    budget = "; ".join(
        f"{r['label']}: max recorded {max(v['validation']['step'] for v in r['vals']):,}"
        for r in runs[:2]
    )
    fig.suptitle(title + "\n" + budget, fontsize=11)
    fig.text(0.5, 0.005, NOTE, ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.075, 1, 0.94))
    fig.savefig(output / name, dpi=135)
    plt.close(fig)


def render(channels, fields, runs, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    temp = [r for r in fields if r["field"] == "thetao"]
    for ax, metric, label in zip(
        axes,
        ("normalized_rmse", "full_grid_nll"),
        (
            "sqrt(mean 19-channel normalized MSE)",
            "Mean hard-label NLL (nats/full-grid cell)",
        ),
        strict=True,
    ):
        plot_panel(ax, temp, runs, metric, "Temperature: all 19 depths", label)
    finish(
        fig,
        axes,
        runs,
        output,
        "temperature-curves.png",
        "Fixed validation samples; raw and EMA temperature curves",
    )
    for metric, name, label in (
        ("normalized_rmse", "temperature-depth-rmse.png", "Normalized full-grid RMSE"),
        (
            "full_grid_nll",
            "temperature-depth-nll.png",
            "Hard NLL (nats/full-grid cell)",
        ),
    ):
        fig, axes = plt.subplots(4, 5, figsize=(15, 11))
        for depth, ax in enumerate(axes.flat):
            if depth == 19:
                ax.set_axis_off()
                continue
            rows = [r for r in channels if r["channel"] == f"thetao_{depth}"]
            plot_panel(ax, rows, runs, metric, f"{rows[0]['depth_m']:g} m", label)
        finish(fig, axes, runs, output, name, "Temperature by depth")
    fig, axes = plt.subplots(2, 5, figsize=(16, 7.5))
    for j, field in enumerate(FIELDS):
        rows = [r for r in fields if r["field"] == field]
        plot_panel(
            axes[0, j], rows, runs, "full_grid_nmse", field, "Normalized full-grid MSE"
        )
        plot_panel(
            axes[1, j],
            rows,
            runs,
            "full_grid_nll",
            field,
            "Hard NLL (nats/full-grid cell)",
        )
    finish(
        fig,
        axes,
        runs,
        output,
        "field-curves.png",
        "Supervised fields only; T-only auxiliary outputs are excluded",
    )


def summary(runs):
    result = {}
    for r in runs:
        logs = [x for x in r["training"] if "elapsed_seconds" in x]
        steady = [x for x in logs if x["elapsed_seconds"] >= 120]
        util = [
            x["utilization"]
            for x in steady
            if isinstance(x.get("utilization"), (int, float))
        ]
        times = [
            x["seconds"] for x in logs if isinstance(x.get("seconds"), (int, float))
        ]
        last = max(logs, key=lambda x: x["step"]) if logs else {}
        cfg = r["provenance"]["config"]
        requested = cfg.get("max_updates", cfg.get("steps"))
        max_val = max(x["validation"]["step"] for x in r["vals"])
        final_ema = any(
            x["weights"] == "ema" and x["validation"]["step"] == requested
            for x in r["vals"]
        )
        result[r["name"]] = dict(
            path=r["path"],
            benchmark=r["benchmark"],
            matched_validation_samples=r["matched_validation_samples"],
            likelihood_comparable=r["likelihood_comparable"],
            complete=r["complete"],
            requested_updates=requested,
            final_requested_ema_available=final_ema,
            max_validation_update=max_val,
            last_logged_update=last.get("step"),
            accepted_examples=last.get("accepted_examples"),
            last_logged_training_elapsed_seconds=last.get("elapsed_seconds"),
            median_logged_update_seconds=float(np.median(times)) if times else None,
            mean_logged_utilization_after120s=float(np.mean(util)) if util else None,
            source_commit=r["provenance"].get("source_commit"),
            job=r["provenance"].get("job"),
            initial_parameter_sha256=r["provenance"].get("initial_parameter_sha256"),
        )
    return dict(
        runs=result,
        notes=[
            "Partial runs are permitted; maximum recorded update is not proof of completion. Final endpoint is requested-update EMA, not best checkpoint.",
            "Logged training elapsed time is not allocated GPU cost; obtain Slurm accounting separately, including failures and retries.",
            "Temperature aggregate RMSE is sqrt(mean normalized channel MSE), not mean of depth RMSEs or physical degrees C.",
            "Step zero is preserved in CSV and omitted from log-axis plots explicitly. All raw/EMA per-channel metrics, including unsupervised outputs, remain in the channel CSV.",
            "Historical curves are benchmarks; matched samples do not match architecture, objective, context, seed or budget. NLL overlays require equal vocabulary/bounds/objective; differing likelihoods remain in CSV only.",
            "Persistence is a point forecast and has no categorical NLL in these receipts.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory containing temperature-both-25k and all-fields-both-25k",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--benchmark",
        action="append",
        default=[],
        metavar="LABEL=DIRECTORY",
        help="Optional historical per-channel validation directory; global-only summaries are unsupported",
    )
    args = parser.parse_args()
    runs = [read_run(args.root / name, name) for name in ARMS]
    for option in args.benchmark:
        if "=" not in option:
            parser.error("--benchmark needs LABEL=DIRECTORY")
        name, path = option.split("=", 1)
        if not name or name in {r["name"] for r in runs}:
            parser.error("Benchmark labels must be nonempty and unique")
        runs.append(read_run(Path(path), name, benchmark=True))
    check_runs(runs)
    channels, fields, training = tables(runs)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "validation-channels.csv.gz", channels)
    write_csv(args.output / "validation-fields.csv", fields)
    write_csv(args.output / "training-progress.csv", training)
    render(channels, fields, runs, args.output)
    (args.output / "curve-receipt.json").write_text(
        json.dumps(summary(runs), indent=2) + "\n"
    )
    with gzip.open(args.output / "curve-provenance.json.gz", "wt") as f:
        json.dump({r["name"]: r["provenance"] for r in runs}, f)


if __name__ == "__main__":
    main()
