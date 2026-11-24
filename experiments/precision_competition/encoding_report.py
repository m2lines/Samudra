# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Encoding-wave exports with paired controls and explicit qualification."""

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np

from .encoding import KINDS

EXTENSION_KINDS = [
    "cont_shrink_mse",
    "cont_shrink_normalized",
    "cont_fixed_mse",
    "ar_overlap_scaled",
    "ar_twohot_hard_scaled",
    "ar_twohot_soft_scaled",
]
ALL_KINDS = list(KINDS) + EXTENSION_KINDS

LABELS = {
    "mse": "MSE",
    "asinh_mse": "asinh MSE",
    "radix": "Parallel radix",
    "parallel_nested": "Parallel nested",
    "ar_uniform": "AR uniform nested",
    "ar_nested": "AR asinh nested",
    "ar_overlap": "AR overlap / hard CE",
    "ar_twohot_hard": "AR overlap / two-hot / hard",
    "ar_twohot_soft": "AR overlap / two-hot / soft",
    "flat_twohot": "Flat two-hot",
    "flat_gauss": "Flat Gaussian histogram",
    "cont_shrink_mse": "Continuous shrink / MSE",
    "cont_shrink_normalized": "Continuous shrink / normalized",
    "cont_fixed_mse": "Continuous fixed range / MSE",
    "ar_overlap_scaled": "AR overlap / scaled hard CE",
    "ar_twohot_hard_scaled": "AR overlap / scaled two-hot / hard",
    "ar_twohot_soft_scaled": "AR overlap / scaled two-hot / soft",
}


def bits(rmse):
    if rmse < 0 or not math.isfinite(rmse):
        raise ValueError("Invalid RMSE")
    return -math.log2(rmse) if rmse else math.inf


def penalty(rmse, reference):
    if rmse == reference == 0:
        return None
    return bits(reference) - bits(rmse)


def phase_control(phase):
    if phase in {"main", "clean", "noisy"}:
        return "main"
    if phase.startswith("probe"):
        return phase.split("-")[0] + "-main"
    return None


def identity(bundle):
    entry = bundle["result"]["metadata"]["entry"]
    return {k: entry[k] for k in ("id", "kind", "lr", "seed")}


def flatten_metrics(row):
    scalar = {}
    for key, value in row.items():
        if key == "bits":
            continue
        if isinstance(value, list):
            scalar.update({f"{key}_{i + 1}": v for i, v in enumerate(value)})
        else:
            scalar[key] = value
    scalar["bits"] = bits(row["rmse"])
    return scalar


def curve_rows(bundle):
    ident = identity(bundle)
    phases = bundle["phases"]
    rows = []
    for phase, values in phases.items():
        history = values["curves"]
        control_name = phase_control(phase)
        control = (
            {row["step"]: row for row in phases[control_name]["curves"]}
            if control_name
            else {}
        )
        for view in ("raw", "ema"):
            initial = history[0][view]["rmse"]
            for row in history:
                current = row[view]
                counterpart = control.get(row["step"])
                rows.append(
                    {
                        **ident,
                        "phase": phase,
                        "view": view,
                        "step": row["step"],
                        "examples": row["step"] * 64,
                        **flatten_metrics(current),
                        "bits_lost_from_start": penalty(current["rmse"], initial),
                        "control_bits_lost": penalty(
                            current["rmse"], counterpart[view]["rmse"]
                        )
                        if counterpart
                        else None,
                        "control_observed": counterpart is not None,
                    }
                )
    return rows


def test_rows(bundle):
    ident = identity(bundle)
    rows = []
    for phase, values in bundle["phases"].items():
        control = phase_control(phase)
        for view in ("raw", "ema"):
            current = values["result"][view]
            reference = bundle["phases"][control]["result"][view] if control else None
            rows.append(
                {
                    **ident,
                    "phase": phase,
                    "view": view,
                    **flatten_metrics(current),
                    "control_bits_lost": penalty(current["rmse"], reference["rmse"])
                    if reference
                    else None,
                }
            )
    return rows


def summarize(rows, keys, metrics):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    result = []
    for group, members in sorted(grouped.items()):
        if len({m["seed"] for m in members}) != len(members):
            raise ValueError("Repeated seed within group")
        item = dict(zip(keys, group, strict=True))
        item["seeds"] = len(members)
        for metric in metrics:
            values = [m[metric] for m in members if m.get(metric) is not None]
            if values:
                item.update(
                    {
                        f"{metric}_median": median(values),
                        f"{metric}_min": min(values),
                        f"{metric}_max": max(values),
                    }
                )
        result.append(item)
    return result


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
    payload = text.getvalue().encode()
    path.write_bytes(
        gzip.compress(payload, mtime=0) if path.suffix == ".gz" else payload
    )


def validate(bundle):
    result = bundle["result"]
    entry = result["metadata"]["entry"]
    if entry.get("smoke") or entry["solo_steps"] != 12000:
        raise ValueError("Smoke or wrong training budget")
    if result["metadata"]["device"] != "NVIDIA L40S":
        raise ValueError("Wrong GPU")
    phases = bundle["phases"]
    if phases["solo"]["curves"][-1]["step"] != 12000:
        raise ValueError("Incomplete solo phase")
    gate = (
        all(r["raw"]["bits"] >= 4 for r in phases["solo"]["curves"][-3:])
        and result["aux_fit"]["qualified"]
    )
    if result["qualified"] != gate:
        raise ValueError("Qualification mismatch")
    expected_forks = {"main", "clean", "noisy"} if gate else set()
    if set(result["forks"]) != expected_forks:
        raise ValueError("Missing or unexpected joint branches")
    for name, phase in phases.items():
        horizon = 12000 if name == "solo" else 256 if name.startswith("probe") else 3000
        if phase["curves"][-1]["step"] != horizon:
            raise ValueError("Incomplete phase")
        for view in ("raw", "ema"):
            for row in phase["curves"] + [phase["result"]]:
                bits(row[view]["rmse"])
        if phase_control(name):
            reference = phases[phase_control(name)]["curves"][0]
            for view in ("raw", "ema"):
                if phase["curves"][0][view]["rmse"] != reference[view]["rmse"]:
                    raise ValueError("Unpaired initial weights")


def actual_data_oracle_audit():
    import torch

    from .benchmark import Config
    from .encoding import EncodedModel, Source, quantize

    torch.set_num_threads(4)
    source = Source(Config(), "cpu")
    rng = source.generator(99001)
    truth = torch.cat([source.draw(64, rng)["main"] for _ in range(8192 // 64)])
    records = []
    for dtype in (torch.float32, torch.float64):
        for kind in KINDS:
            model = EncodedModel(source.c, source.positions, kind).to(dtype=dtype)
            predictions = []
            with torch.no_grad():
                for values in truth.to(dtype).split(256):
                    if kind == "mse":
                        prediction = values
                    elif kind == "asinh_mse":
                        prediction = values.asinh().sinh()
                    else:
                        prediction = model.decode(
                            torch.zeros(len(values), 128, dtype=dtype),
                            values,
                            oracle=True,
                        )[0]
                    predictions.append(prediction)
            predicted = torch.cat(predictions)
            error = predicted.double() - truth.double()
            rmse = error.square().mean().sqrt().item()
            records.append(
                {
                    "kind": kind,
                    "dtype": str(dtype),
                    "rmse": rmse,
                    "bits": bits(rmse),
                    "max_abs": error.abs().max().item(),
                    "exact_values": (error == 0).sum().item(),
                    "rounded_words": (quantize(predicted)[1] == quantize(truth)[1])
                    .sum()
                    .item(),
                    "examples": len(truth),
                }
            )
    return records


def active_kinds(rows):
    return [k for k in ALL_KINDS if any(r["kind"] == k for r in rows)]


def plot_learning(rows, output):
    import matplotlib.pyplot as plt

    kinds = active_kinds(rows)
    nrows = math.ceil((len(kinds) + 1) / 3)
    fig, axes = plt.subplots(
        nrows, 3, figsize=(15, 3.5 * nrows), sharex=True, squeeze=False
    )
    for ax, kind in zip(axes.flat, kinds, strict=False):
        for lr, color in [(1e-4, "#2166ac"), (1e-5, "#b2182b")]:
            for view, style in [("raw", "-"), ("ema", "--")]:
                selected = [
                    r
                    for r in rows
                    if r["kind"] == kind
                    and r["phase"] == "solo"
                    and r["lr"] == lr
                    and r["view"] == view
                ]
                summary = summarize(selected, ["step"], ["bits"])
                ax.plot(
                    [r["step"] for r in summary],
                    [r["bits_median"] for r in summary],
                    style,
                    color=color,
                    label=f"{lr:g}, {view}",
                )
                if view == "raw":
                    ax.fill_between(
                        [r["step"] for r in summary],
                        [r["bits_min"] for r in summary],
                        [r["bits_max"] for r in summary],
                        color=color,
                        alpha=0.12,
                    )
        ax.axhline(4, color="gray", lw=0.8)
        ax.set(title=LABELS[kind], ylabel="RMS bits (higher is better)")
        ax.grid(alpha=0.2)
    for ax in list(axes.flat)[len(kinds) :]:
        ax.axis("off")
    axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(), loc="center")
    for ax in axes[-1]:
        ax.set_xlabel("Solo updates")
    fig.suptitle(
        "Learning the copy task: seed median; shading is raw seed range; dashed = EMA 0.99"
    )
    fig.tight_layout()
    fig.savefig(output / "copy-learning.png", dpi=150)
    plt.close(fig)


def plot_joint(rows, output):
    import matplotlib.pyplot as plt

    kinds = active_kinds(rows)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    for column, lr in enumerate((1e-4, 1e-5)):
        for row, view in enumerate(("raw", "ema")):
            ax = axes[row, column]
            for offset, phase, color in [
                (-0.13, "clean", "#2166ac"),
                (0.13, "noisy", "#b2182b"),
            ]:
                selected = [
                    r
                    for r in rows
                    if r["lr"] == lr and r["view"] == view and r["phase"] == phase
                ]
                for item in summarize(selected, ["kind"], ["control_bits_lost"]):
                    x = kinds.index(item["kind"]) + offset
                    m, lo, hi = (
                        item["control_bits_lost_" + stat]
                        for stat in ("median", "min", "max")
                    )
                    ax.errorbar(
                        x, m, yerr=[[m - lo], [hi - m]], fmt="o", color=color, capsize=3
                    )
                    ax.text(
                        x,
                        hi + 0.05,
                        str(item["seeds"]),
                        fontsize=7,
                        ha="center",
                        color=color,
                    )
                ax.plot([], [], "o", color=color, label=phase)
            ax.axhline(0, color="gray", lw=0.8)
            ax.set(
                title=f"LR {lr:g}, {view}",
                ylabel="Bits lost versus paired main-only (lower is better)",
            )
            ax.set_xticks(
                range(len(kinds)), [LABELS[k] for k in kinds], rotation=60, ha="right"
            )
            ax.grid(axis="y", alpha=0.2)
            ax.legend()
    fig.suptitle(
        "Joint training: independent final test; median and seed range; numbers = qualified seeds"
    )
    fig.tight_layout()
    fig.savefig(output / "joint-interference.png", dpi=150)
    plt.close(fig)


def plot_stages(rows, output):
    import matplotlib.pyplot as plt

    kinds = [
        k
        for k in active_kinds(rows)
        if k.startswith(("ar_", "cont_")) or k == "parallel_nested"
    ]
    nrows = math.ceil(len(kinds) / 3)
    fig, axes = plt.subplots(nrows, 3, figsize=(15, 4.5 * nrows), squeeze=False)
    for ax, kind in zip(axes.flat, kinds, strict=False):
        for lr, color in [(1e-4, "#2166ac"), (1e-5, "#b2182b")]:
            selected = [
                r
                for r in rows
                if r["kind"] == kind
                and r["phase"] == "solo"
                and r["lr"] == lr
                and r["view"] == "raw"
            ]
            for prefix, style in [
                ("stage_rmse", "-"),
                ("stage_rmse_first_wrong", "--"),
            ]:
                y = [
                    median(
                        [
                            r[f"{prefix}_{i}"]
                            for r in selected
                            if r.get(f"{prefix}_{i}") is not None
                        ]
                    )
                    if any(r.get(f"{prefix}_{i}") is not None for r in selected)
                    else np.nan
                    for i in range(1, 5)
                ]
                ax.plot(
                    range(1, 5),
                    y,
                    style,
                    color=color,
                    marker="o",
                    label=f"{lr:g}, {'all' if style == '-' else 'first wrong'}",
                )
        ax.set(
            title=LABELS[kind],
            xlabel="Refinement stage",
            ylabel="Normalized RMSE",
            yscale="log",
            xticks=range(1, 5),
        )
        ax.grid(alpha=0.2)
    for ax in list(axes.flat)[len(kinds) :]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(
        "Final solo test: stage error; dashed curves condition on an incorrect first argmax"
    )
    fig.tight_layout()
    fig.savefig(output / "refinement-stages.png", dpi=150)
    plt.close(fig)


def plot_auxiliary(rows, output):
    import matplotlib.pyplot as plt

    kinds = active_kinds(rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, lr in zip(axes, (1e-4, 1e-5), strict=True):
        for phase, marker, offset in [("clean", "o", -0.12), ("noisy", "s", 0.12)]:
            selected = [
                r
                for r in rows
                if r["phase"] == phase and r["lr"] == lr and r["view"] == "raw"
            ]
            for item in summarize(selected, ["kind"], ["aux_nrmse"]):
                m, lo, hi = (
                    item["aux_nrmse_" + stat] for stat in ("median", "min", "max")
                )
                ax.errorbar(
                    kinds.index(item["kind"]) + offset,
                    m,
                    yerr=[[m - lo], [hi - m]],
                    fmt=marker,
                    color="#2166ac" if phase == "clean" else "#b2182b",
                    capsize=3,
                )
            ax.plot(
                [],
                [],
                marker,
                label=phase,
                color="#2166ac" if phase == "clean" else "#b2182b",
            )
        ax.axhline(1, color="gray", ls="--", label="Constant predictor")
        ax.set(
            title=f"LR {lr:g}", ylabel="Clean auxiliary normalized RMSE", yscale="log"
        )
        ax.set_xticks(
            range(len(kinds)), [LABELS[k] for k in kinds], rotation=60, ha="right"
        )
        ax.legend()
    fig.suptitle(
        "Auxiliary accuracy after joint training: raw weights, independent test, seed median/range"
    )
    fig.tight_layout()
    fig.savefig(output / "auxiliary-accuracy.png", dpi=150)
    plt.close(fig)


def plot_probe(rows, output):
    import matplotlib.pyplot as plt

    kinds = active_kinds(rows)
    colors = plt.get_cmap("tab20")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for column, lr in enumerate((1e-4, 1e-5)):
        for row, level in enumerate((4, 6)):
            ax = axes[row, column]
            for kind in kinds:
                for condition, style in [("clean", "-"), ("noisy", "--")]:
                    selected = [
                        r
                        for r in rows
                        if r["kind"] == kind
                        and r["lr"] == lr
                        and r["view"] == "raw"
                        and r["phase"] == f"probe{level}-{condition}"
                    ]
                    grouped = summarize(selected, ["step"], ["bits_lost_from_start"])
                    if grouped:
                        ax.plot(
                            [r["step"] for r in grouped],
                            [r["bits_lost_from_start_median"] for r in grouped],
                            style,
                            color=colors(kinds.index(kind)),
                            label=LABELS[kind] if condition == "clean" else None,
                        )
            ax.axhline(0, color="gray", lw=0.8)
            ax.set(
                title=f"Initial threshold {level} bits; LR {lr:g}",
                xlabel="Auxiliary-only updates",
                ylabel="Bits lost from starting checkpoint",
                xscale="symlog",
                xlim=(0, 256),
            )
            ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8)
    fig.suptitle(
        "Direct auxiliary perturbations: seed median, solid clean / dashed noisy; actual starting bits vary"
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(output / "auxiliary-probes.png", dpi=150)
    plt.close(fig)


def plot_aux_learning(rows, output):
    import matplotlib.pyplot as plt

    kinds = active_kinds(rows)
    colors = plt.get_cmap("tab20")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for column, lr in enumerate((1e-4, 1e-5)):
        for row, phase in enumerate(("clean", "noisy")):
            ax = axes[row, column]
            for kind in kinds:
                selected = [
                    r
                    for r in rows
                    if r["kind"] == kind
                    and r["lr"] == lr
                    and r["view"] == "raw"
                    and r["phase"] == phase
                ]
                grouped = summarize(selected, ["step"], ["aux_nrmse"])
                if grouped:
                    ax.plot(
                        [r["step"] for r in grouped],
                        [r["aux_nrmse_median"] for r in grouped],
                        color=colors(kinds.index(kind)),
                        label=LABELS[kind],
                    )
            ax.set(
                title=f"{phase} joint; LR {lr:g}",
                xlabel="Joint updates",
                ylabel="Clean auxiliary normalized RMSE",
                yscale="log",
            )
            ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8)
    fig.suptitle(
        "Auxiliary learning: clean target error, raw weights, median across qualified seeds"
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(output / "auxiliary-learning.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    bundles = json.loads(gzip.decompress(args.input.read_bytes()))
    curves, tests, qualifications, gradients = [], [], [], []
    for bundle in bundles:
        validate(bundle)
        ident = identity(bundle)
        result = bundle["result"]
        qualifications.append(
            {
                **ident,
                "qualified": result["qualified"],
                "copy_qualified": all(
                    row["raw"]["bits"] >= 4
                    for row in bundle["phases"]["solo"]["curves"][-3:]
                ),
                "copy_final_window_min_bits": min(
                    row["raw"]["bits"]
                    for row in bundle["phases"]["solo"]["curves"][-3:]
                ),
                "aux_qualified": result["aux_fit"]["qualified"],
                "aux_fit_nrmse": result["aux_fit"]["metrics"]["aux_nrmse"],
                "aux_head_norm": result["aux_fit"]["conditioning"]["weight_norm"],
                "main_loss_weight": result["metadata"]["weight"],
                "parameters": result["metadata"]["parameters"],
                "decoder_parameters": result["metadata"]["decoder_parameters"],
            }
        )
        curves.extend(curve_rows(bundle))
        tests.extend(test_rows(bundle))
        for phase, values in bundle["phases"].items():
            for row in values["curves"]:
                if "gradients" in row:
                    gradients.append(
                        {
                            **ident,
                            "phase": phase,
                            "step": row["step"],
                            **{
                                f"{k}_{i + 1}": v
                                for k, a in row["gradients"].items()
                                for i, v in enumerate(a)
                            },
                        }
                    )
    if len({r["id"] for r in qualifications}) != len(qualifications):
        raise ValueError("Duplicate bundle")
    write_csv(args.output / "qualification.csv", qualifications)
    write_csv(args.output / "test-metrics.csv", tests)
    write_csv(args.output / "gradients.csv.gz", gradients)
    write_csv(
        args.output / "test-summary.csv",
        summarize(
            tests,
            ["kind", "lr", "phase", "view"],
            [
                "bits",
                "rmse",
                "aux_nrmse",
                "control_bits_lost",
                "p99_abs",
                "rounded_word_accuracy",
            ],
        ),
    )
    for kind in active_kinds(curves):
        write_csv(
            args.output / f"curves-{kind}.csv.gz",
            [r for r in curves if r["kind"] == kind],
        )
        payload = json.dumps(
            [b for b in bundles if identity(b)["kind"] == kind], separators=(",", ":")
        ).encode()
        (args.output / f"raw-{kind}.json.gz").write_bytes(
            gzip.compress(payload, mtime=0)
        )
    write_csv(args.output / "oracle-test-distribution.csv", actual_data_oracle_audit())
    plot_learning(curves, args.output)
    plot_joint(tests, args.output)
    plot_stages(tests, args.output)
    plot_auxiliary(tests, args.output)
    plot_probe(curves, args.output)
    plot_aux_learning(curves, args.output)
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(args.output.iterdir())
        if p.is_file() and p.name != "sha256.json"
    }
    (args.output / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")


if __name__ == "__main__":
    main()
