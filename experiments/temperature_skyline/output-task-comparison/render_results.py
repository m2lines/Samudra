# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Render fixed-step raw/EMA validation comparisons from saved JSON receipts."""

import argparse
import csv
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARMS = ("temperature", "all_fields", "all_depths")
LABELS = ("T only", "All fields, one depth", "All fields, all depths")
COLORS = ("#0072b2", "#d55e00", "#009e73")
FIELDS = {
    "thetao": ("Temperature", "°C"),
    "so": ("Salinity", "psu"),
    "uo": ("Zonal velocity", "m/s"),
    "vo": ("Meridional velocity", "m/s"),
    "zos": ("SSH", "m"),
}


def collect(root):
    records = []
    finals: dict[str, dict] = {}
    provenance = {}
    for arm in ARMS:
        folder = root / f"{arm}-25k"
        provenance[arm] = json.loads((folder / "provenance-restart-0.json").read_text())
        finals[arm] = {}
        for path in sorted(folder.glob("validation-*.json")):
            kind = path.name.split("-")[1]
            if kind not in {"raw", "ema"}:
                continue
            result = json.loads(path.read_text())
            step = result["step"]
            for row in result["per_channel"]:
                records.append(dict(arm=arm, weights=kind, step=step, **row))
            if step == 25000:
                finals[arm][kind] = result
    initial = {p["initial_parameter_sha256"] for p in provenance.values()}
    assert len(initial) == 1, "Initial parameters differ"
    return records, finals, provenance


def channel_series(records, arm, kind, channel, key, minimum):
    rows = sorted(
        (
            r
            for r in records
            if r["arm"] == arm
            and r["weights"] == kind
            and r["channel"] == channel
            and r["step"] >= minimum
        ),
        key=lambda r: r["step"],
    )
    return [r["step"] for r in rows], [r[key] for r in rows]


def axis_style(ax, minimum):
    if minimum == 0:
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xticks([0, 1, 1000, 25000], ["0", "1", "1k", "25k"])
    else:
        ax.set_xlim(minimum, 25000)
        ax.set_xticks([1000, 10000, 25000], ["1k", "10k", "25k"])
    ax.set_yscale("log")
    ax.grid(alpha=0.18, which="major")
    ax.tick_params(labelsize=8)
    ax.spines[["right", "top"]].set_visible(False)


def render(records, finals, output):
    for minimum, suffix in ((0, "full"), (1000, "after-1k")):
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
        for ax, field in zip(axes.flat, FIELDS, strict=False):
            for arm, label, color in zip(ARMS, LABELS, COLORS, strict=True):
                for kind, style in (("raw", "--"), ("ema", "-")):
                    rows = [
                        r
                        for r in records
                        if r["arm"] == arm
                        and r["weights"] == kind
                        and r["channel"].split("_")[0] == field
                        and r["step"] >= minimum
                    ]
                    steps = sorted({r["step"] for r in rows})
                    values = [
                        np.mean(
                            [r["full_grid_nmse"] for r in rows if r["step"] == step]
                        )
                        for step in steps
                    ]
                    if steps:
                        ax.plot(
                            steps,
                            values,
                            color=color,
                            linestyle=style,
                            linewidth=1.4,
                            alpha=0.55 if kind == "raw" else 1,
                            label=f"{label}: {kind}",
                        )
            ax.set_title(FIELDS[field][0])
            axis_style(ax, minimum)
            ax.set_ylabel("Mean-channel normalized MSE")
            ax.set_xlabel("Optimizer updates")
        axes.flat[-1].axis("off")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        axes.flat[-1].legend(handles, labels, loc="center", frameon=False)
        fig.suptitle(
            "Output-task comparison · fixed validation cases · legacy full-grid MSE\nSolid: EMA; dashed: raw · lower is better",
            fontsize=12,
        )
        fig.savefig(output / f"validation-fields-{suffix}.png", dpi=150)
        plt.close(fig)

    for field, (title, unit) in FIELDS.items():
        if field == "zos":
            continue
        fig, axes = plt.subplots(4, 5, figsize=(16, 10), layout="constrained")
        for z, ax in enumerate(axes.flat[:19]):
            channel = f"{field}_{z}"
            depth = next(r["depth_m"] for r in records if r["channel"] == channel)
            for arm, label, color in zip(ARMS, LABELS, COLORS, strict=True):
                for kind, style in (("raw", "--"), ("ema", "-")):
                    steps, values = channel_series(
                        records, arm, kind, channel, "wet_cell_rmse", 1000
                    )
                    if steps:
                        ax.plot(
                            steps,
                            values,
                            color=color,
                            linestyle=style,
                            alpha=0.5 if kind == "raw" else 1,
                            linewidth=1.2,
                            label=f"{label}: {kind}",
                        )
            rows = [r for r in records if r["channel"] == channel and r["step"] == 0]
            if rows:
                r = rows[0]
                scale_ratio = r["wet_cell_rmse"] ** 2 / r["full_grid_nmse"]
                persist = np.sqrt(r["persistence_full_grid_nmse"] * scale_ratio)
                ax.axhline(
                    persist, color=".5", linestyle=":", linewidth=1, label="Persistence"
                )
            ax.set_title(f"{depth:g} m", fontsize=10)
            axis_style(ax, 1000)
            ax.set_ylabel(f"RMSE ({unit})", fontsize=8)
        axes.flat[-1].axis("off")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        axes.flat[-1].legend(handles, labels, loc="center", frameon=False, fontsize=8)
        fig.suptitle(
            f"{title}: wet-cell RMSE by depth · fixed validation cases · updates 1,000–25,000\nSolid: EMA; dashed: raw; dotted: persistence · lower is better",
            fontsize=12,
        )
        fig.supxlabel("Optimizer updates")
        fig.savefig(output / f"validation-{field}-depths.png", dpi=150)
        plt.close(fig)

    # Final-step depth profiles only exist once every arm has completed.
    if all(set(finals[arm]) == {"raw", "ema"} for arm in ARMS):
        fig, axes = plt.subplots(1, 4, figsize=(14, 6), layout="constrained")
        for ax, field in zip(axes, ("thetao", "so", "uo", "vo"), strict=True):
            for arm, label, color in zip(ARMS, LABELS, COLORS, strict=True):
                for kind, style in (("raw", "--"), ("ema", "-")):
                    rows = [
                        r
                        for r in finals[arm][kind]["per_channel"]
                        if r["channel"].startswith(field + "_")
                    ]
                    if rows:
                        ax.plot(
                            [r["wet_cell_rmse"] for r in rows],
                            [r["depth_m"] for r in rows],
                            color=color,
                            linestyle=style,
                            alpha=0.55 if kind == "raw" else 1,
                            label=f"{label}: {kind}",
                        )
            ax.set_title(FIELDS[field][0])
            ax.set_xlabel(f"Wet-cell RMSE ({FIELDS[field][1]})")
            ax.set_xscale("log")
            ax.set_yscale("symlog", linthresh=10)
            ax.invert_yaxis()
            ax.set_ylabel("Depth (m)")
            ax.grid(alpha=0.2)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
        fig.suptitle(
            "Final update 25,000 · fixed validation cases · raw and EMA weights"
        )
        fig.savefig(output / "final-depth-profiles.png", dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records, finals, provenance = collect(args.source)
    with gzip.open(args.output / "validation-per-channel.csv.gz", "wt") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = dict(
        final_update=25000,
        final=finals,
        initial_parameter_sha256=next(iter(provenance.values()))[
            "initial_parameter_sha256"
        ],
        jobs={arm: p["job"] for arm, p in provenance.items()},
        source_commit=next(iter(provenance.values()))["source_commit"],
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    render(records, finals, args.output)


if __name__ == "__main__":
    main()
