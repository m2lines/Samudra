# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Paired noise-sweep exports; all contrasts use observed common checkpoints."""

import argparse
import csv
import gzip
import io
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from .report import write_csv

VIEWS = ("raw", "0.99", "0.999")
CONDITIONS = [
    ("unlearned", 0.0),
    ("learned", 0.0),
    ("learned", 0.1),
    ("learned", 0.3),
    ("learned", 1.0),
]


def identity(entry):
    parts = entry["id"].split("-")
    return dict(
        stage=entry["stage"],
        id=entry["id"],
        seed=int(parts[0][4:]),
        level=int(parts[2]),
        lr=entry.get("lr"),
        batch=entry.get("batch"),
    )


def metric(row, view):
    return row["raw"] if view == "raw" else row["ema"][view]


def probe_decomposition(row):
    d = np.asarray(row["displacements"], dtype=float)
    if d.shape != (16, 64):
        raise ValueError("Expected 16 probe batches and 64 held-out outputs")
    mean_squared = np.square(d.mean(axis=0)).mean()
    variance = d.var(axis=0, ddof=1).mean()
    second_moment = np.square(d).mean()
    np.testing.assert_allclose(
        second_moment, mean_squared + 15 / 16 * variance, rtol=1e-12, atol=1e-20
    )
    np.testing.assert_allclose(
        row["mean_mse_change"],
        row["error_alignment_term"] + second_moment,
        rtol=1e-12,
        atol=1e-20,
    )
    return dict(
        mean_displacement_squared=float(mean_squared),
        displacement_second_moment=float(second_moment),
        corrected_squared_mean=float(mean_squared - variance / 16),
    )


def trajectory_rows(bundle):
    ident = identity(bundle["entry"])
    control = next(t for t in bundle["trajectories"] if t["condition"] == "main")
    control_history = {r["step"]: r for r in control["history"]}
    rows = []
    for trajectory in bundle["trajectories"]:
        for view in VIEWS:
            start = metric(trajectory["history"][0], view)["bits"]
            for row in trajectory["history"]:
                step = row["step"]
                reference = metric(control_history[step], view)
                values = metric(row, view)
                recovery = ident["stage"] == "perturb" and step > 256
                phase_step = step - 256 if recovery else step
                rows.append(
                    {
                        **ident,
                        "condition": trajectory["condition"],
                        "sigma": trajectory["sigma"],
                        "view": view,
                        "step": step,
                        "phase": "recovery" if recovery else ident["stage"],
                        "phase_step": phase_step,
                        "examples": row["examples"],
                        "phase_examples": phase_step * ident["batch"],
                        **values,
                        "starting_bits": start,
                        "bits_lost_from_start": start - values["bits"],
                        "control_bits_lost": reference["bits"] - values["bits"],
                    }
                )
    return rows


def summarize(rows, keys, metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    summaries = []
    for key, group in groups.items():
        seeds = [r["seed"] for r in group]
        if len(seeds) != len(set(seeds)):
            raise ValueError("Repeated seed within a comparison")
        result = dict(zip(keys, key, strict=True))
        result["seeds"] = len(seeds)
        for name in metrics:
            values = [r[name] for r in group]
            result.update(
                {
                    name + "_median": median(values),
                    name + "_min": min(values),
                    name + "_max": max(values),
                }
            )
        summaries.append(result)
    return summaries


def write_gzip_csv(path, rows):
    stream = io.StringIO()
    writer = csv.DictWriter(
        stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r))
    )
    writer.writeheader()
    writer.writerows(rows)
    path.write_bytes(gzip.compress(stream.getvalue().encode(), mtime=0))


def exports(bundles, output):
    trajectories, probes, endpoints = [], [], []
    for bundle in bundles:
        if bundle["entry"]["stage"] == "prepare":
            continue
        ident = identity(bundle["entry"])
        trajectories.extend(trajectory_rows(bundle))
        control = next(t for t in bundle["trajectories"] if t["condition"] == "main")
        for t in bundle["trajectories"]:
            for view in VIEWS:
                values = t["test"] if view == "raw" else t["ema_test"][view]
                reference = (
                    control["test"] if view == "raw" else control["ema_test"][view]
                )
                endpoints.append(
                    {
                        **ident,
                        "condition": t["condition"],
                        "sigma": t["sigma"],
                        "view": view,
                        "endpoint": t["steps"] + t["recovery_steps"],
                        "training_seconds": t["training_seconds"],
                        "test_phase": "post_recovery"
                        if t["recovery_steps"]
                        else "joint_endpoint",
                        "control_bits_lost": reference["bits"] - values["bits"],
                        **values,
                    }
                )
        for probe in bundle.get("probes", []) + (
            [
                {
                    "condition": "coast",
                    "sigma": 0.0,
                    "measurements": bundle["coasting_probes"],
                }
            ]
            if "coasting_probes" in bundle
            else []
        ):
            for row in probe["measurements"]:
                probes.append(
                    {
                        **ident,
                        "condition": probe["condition"],
                        "sigma": probe["sigma"],
                        **{k: v for k, v in row.items() if k != "displacements"},
                        **probe_decomposition(row),
                    }
                )
    write_gzip_csv(output / "trajectories.csv.gz", trajectories)
    write_csv(output / "test-endpoints.csv", endpoints)
    write_csv(
        output / "test-summary.csv",
        summarize(
            endpoints,
            [
                "stage",
                "level",
                "lr",
                "batch",
                "condition",
                "sigma",
                "view",
                "endpoint",
                "test_phase",
            ],
            ["bits", "control_bits_lost", "aux_clean_nrmse", "rmse", "bias"],
        ),
    )
    write_csv(output / "probes.csv", probes)
    keys = ["stage", "level", "lr", "batch", "condition", "sigma", "view", "step"]
    summary = summarize(
        trajectories,
        keys,
        [
            "bits",
            "bits_lost_from_start",
            "control_bits_lost",
            "aux_clean_nrmse",
            "rmse",
            "bias",
        ],
    )
    write_gzip_csv(output / "trajectory-summary.csv.gz", summary)
    write_csv(
        output / "probe-summary.csv",
        summarize(
            probes,
            ["level", "lr", "batch", "condition", "sigma", "optimizer"],
            [
                "mean_displacement_rms",
                "displacement_variance",
                "error_alignment_term",
                "mean_mse_change",
                "corrected_squared_mean",
            ],
        ),
    )
    matched = [
        r
        for r in trajectories
        if (r["stage"] == "perturb" and r["phase_examples"] in (4096, 16384))
        or (r["stage"] == "joint" and r["phase_examples"] in (65536, 131072))
    ]
    write_gzip_csv(output / "matched-examples.csv.gz", matched)
    # Descriptive matching only: first observed point within a fixed 10% target band.
    # Missing bands stay missing; there is no interpolation or extrapolation.
    groups = defaultdict(list)
    for row in trajectories:
        if (
            row["view"] == "raw"
            and row["phase"] == "perturb"
            and row["step"] > 0
            and row["condition"] not in ("main", "offset")
        ):
            groups[(row["id"], row["condition"], row["sigma"])].append(row)
    matched_aux = []
    for group in groups.values():
        for target in (0.05, 0.1, 0.2, 0.5, 1.0):
            candidates = [
                r for r in group if 0.9 * target <= r["aux_clean_nrmse"] <= 1.1 * target
            ]
            if candidates:
                row = min(candidates, key=lambda r: r["step"])
                matched_aux.append({**row, "target_aux_nrmse": target})
    write_csv(output / "matched-auxiliary.csv", matched_aux)
    return trajectories, probes


def figures(rows, probes, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def line(ax, selected, xkey, ykey, label, color, style="-"):
        x = sorted({r[xkey] for r in selected})
        groups = [[r[ykey] for r in selected if r[xkey] == v] for v in x]
        if not x:
            return
        assert all(len(g) == 3 for g in groups)
        ax.plot(
            x, [median(g) for g in groups], label=label, color=color, linestyle=style
        )
        ax.fill_between(
            x,
            [min(g) for g in groups],
            [max(g) for g in groups],
            alpha=0.12,
            color=color,
        )

    shown = [
        ("unlearned", 0.0, "Unlearned clean"),
        ("learned", 0.0, "Fitted clean"),
        ("learned", 1.0, "Fitted noisy, sigma 1"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (level, lr) in zip(
        axes.flat, [(4, 3e-4), (4, 1e-5), (6, 3e-4), (6, 1e-5)], strict=True
    ):
        for i, (condition, sigma, label) in enumerate(shown):
            selected = [
                r
                for r in rows
                if r["stage"] == "perturb"
                and r["level"] == level
                and r["lr"] == lr
                and r["batch"] == 64
                and r["condition"] == condition
                and r["sigma"] == sigma
                and r["view"] == "raw"
            ]
            line(ax, selected, "step", "bits_lost_from_start", label, f"C{i}")
        ax.axvline(256, color="black", linestyle=":", label="Main-only recovery starts")
        ax.axhline(0, color="grey", linewidth=0.6)
        ax.set(
            title=f"Raw {level}-bit parents; LR {lr:g}, batch 64",
            xlabel="Total updates",
            ylabel="Bits lost from prepared start",
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Auxiliary-only intervention then recovery: three-seed median and range"
    )
    fig.savefig(output / "perturb-recovery.png", dpi=90)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for i, (condition, sigma, label) in enumerate(shown):
        selected = [
            r
            for r in rows
            if r["stage"] == "perturb"
            and r["level"] == 6
            and r["step"] == 256
            and r["batch"] == 64
            and r["view"] == "raw"
            and r["condition"] == condition
            and r["sigma"] == sigma
        ]
        line(axes[0], selected, "lr", "control_bits_lost", label, f"C{i}")
        selected = [
            r
            for r in rows
            if r["stage"] == "perturb"
            and r["level"] == 6
            and r["step"] == 256
            and r["lr"] == 3e-4
            and r["view"] == "raw"
            and r["condition"] == condition
            and r["sigma"] == sigma
        ]
        line(axes[1], selected, "batch", "control_bits_lost", label, f"C{i}")
    axes[0].set(xscale="log", xlabel="Learning rate, batch 64")
    axes[1].set(xscale="log", xlabel="Batch size, LR 0.0003")
    for ax in axes:
        ax.set(ylabel="Bits lost versus main-only at update 256")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Raw 6-bit parents; equal updates, three-seed medians and ranges")
    fig.savefig(output / "lr-and-batch.png", dpi=100)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, budget in zip(axes, (4096, 16384), strict=True):
        for i, (condition, sigma, label) in enumerate(shown):
            selected = [
                r
                for r in rows
                if r["stage"] == "perturb"
                and r["phase"] == "perturb"
                and r["level"] == 6
                and r["lr"] == 3e-4
                and r["examples"] == budget
                and r["view"] == "raw"
                and r["condition"] == condition
                and r["sigma"] == sigma
            ]
            line(ax, selected, "batch", "control_bits_lost", label, f"C{i}")
        ax.set(
            title=f"{budget:,} training examples",
            xscale="log",
            xlabel="Batch size",
            ylabel="Bits lost versus paired main-only",
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Raw 6-bit parents, LR 0.0003: exact common exposure checkpoints")
    fig.savefig(output / "matched-exposure.png", dpi=100)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, lr in zip(axes, (3e-4, 1e-5), strict=True):
        for i, (condition, sigma, label) in enumerate(shown):
            selected = [
                r
                for r in rows
                if r["stage"] == "perturb"
                and r["phase"] == "perturb"
                and r["level"] == 6
                and r["lr"] == lr
                and r["batch"] == 64
                and r["view"] == "raw"
                and r["condition"] == condition
                and r["sigma"] == sigma
            ]
            line(ax, selected, "step", "aux_clean_nrmse", label, f"C{i}")
        ax.set(
            title=f"LR {lr:g}, batch 64",
            yscale="log",
            xlabel="Auxiliary-only updates",
            ylabel="Clean auxiliary normalized RMSE",
        )
        ax.axhline(
            1, color="grey", linestyle=":", label="Constant 0.1 population baseline"
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Raw 6-bit parents: auxiliary learning and noise; three-seed median and range"
    )
    fig.savefig(output / "auxiliary-learning.png", dpi=100)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for j, opt in enumerate(("sgd", "adam", "inherited_adam")):
        for i, (condition, sigma, label) in enumerate(shown):
            selected = [
                r
                for r in probes
                if r["level"] == 6
                and r["lr"] == 3e-4
                and r["condition"] == condition
                and r["sigma"] == sigma
                and r["optimizer"] == opt
            ]
            line(axes[0, j], selected, "batch", "mean_displacement_rms", label, f"C{i}")
            line(axes[1, j], selected, "batch", "displacement_variance", label, f"C{i}")
        axes[0, j].set(title=opt)
        for ax in axes[:, j]:
            ax.set(xscale="log", yscale="log", xlabel="Batch size")
            ax.grid(alpha=0.2)
        axes[0, j].legend(fontsize=7)
    axes[0, 0].set(ylabel="RMS of sample mean output displacement")
    axes[1, 0].set(ylabel="Mean per-output sample variance")
    fig.suptitle(
        "Frozen raw 6-bit parents, LR 0.0003: 16 batches per model; three model seeds"
    )
    fig.savefig(output / "frozen-probes.png", dpi=90)
    plt.close(fig)

    if any(r["stage"] == "joint" for r in rows):
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
        settings = [
            (3e-4, 64),
            (1e-4, 64),
            (3e-5, 64),
            (1e-5, 64),
            (3e-4, 256),
            (3e-4, 1024),
        ]
        for ax, (lr, batch) in zip(axes.flat, settings, strict=True):
            for sigma, color in [(0.0, "C0"), (1.0, "C1")]:
                for view, style in [("raw", "-"), ("0.99", "--")]:
                    selected = [
                        r
                        for r in rows
                        if r["stage"] == "joint"
                        and r["lr"] == lr
                        and r["batch"] == batch
                        and r["condition"] == "learned"
                        and r["sigma"] == sigma
                        and r["view"] == view
                    ]
                    line(
                        ax,
                        selected,
                        "step",
                        "control_bits_lost",
                        f"sigma {sigma:g}, {'raw' if view == 'raw' else 'EMA ' + view}",
                        color,
                        style,
                    )
            ax.axhline(0, color="grey", linewidth=0.6)
            ax.set(
                title=f"LR {lr:g}, batch {batch}",
                xlabel="Joint updates",
                ylabel="Bits lost versus main-only",
            )
            ax.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(
            "Joint training from raw 6-bit parents; seed-paired medians and ranges"
        )
        fig.savefig(output / "joint-training.png", dpi=90)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.source:
        bundles: list[dict[str, Any]] = []
        for stage in ("prepare", "perturb", "joint"):
            bundles.extend(
                json.loads(p.read_text())
                for p in sorted((args.source / stage).glob("*/result.json"))
            )
        groups = defaultdict(list)
        for b in bundles:
            ident = identity(b["entry"])
            groups[
                f"{ident['stage']}-bits{ident['level']}-seed{ident['seed']}.json.gz"
            ].append(b)
        for name, group in groups.items():
            (args.output / name).write_bytes(
                gzip.compress(json.dumps(group, allow_nan=False).encode(), mtime=0)
            )
    else:
        bundles = [
            b
            for p in sorted(args.output.glob("*-bits*-seed*.json.gz"))
            for b in json.loads(gzip.decompress(p.read_bytes()))
        ]
    rows, probes = exports(bundles, args.output)
    figures(rows, probes, args.output)


if __name__ == "__main__":
    main()
