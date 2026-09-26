# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Render fixed-example A/B maps and aggregate native controls from exported reports."""

import argparse
import csv
import json
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

RUNS = ("A-1729", "A-1730", "B-1729", "B-1730")
VARIABLES = ("thetao", "so", "uo", "vo")


def save_png(figure, path, *, dpi):
    """Lossless RGB PNG removes an unused alpha plane; preserve native pixel sizes."""
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi)
    buffer.seek(0)
    with Image.open(buffer) as rendered:
        rendered.convert("RGB").save(path, optimize=True, compress_level=9)


def native_controls(root, output):
    sums: dict[tuple[str, str, str, str, int, str], list[float]] = defaultdict(
        lambda: [0.0, 0.0]
    )
    for run in RUNS:
        for phase in ("om4", "observation"):
            with (root / run / phase / "metrics.csv").open() as stream:
                for row in csv.DictReader(stream):
                    if row["channel"] in ("thetao_0", "zos"):
                        continue
                    # First pool origins per channel with their wet-area support.
                    key = (
                        run,
                        phase,
                        row["region"],
                        row["mode"],
                        int(row["lead_days"]),
                        row["channel"],
                    )
                    weight = float(row["wet_area"])
                    sums[key][0] += weight * float(row["physical_mse"])
                    sums[key][1] += weight
    channel_groups = defaultdict(list)
    for key, (total, weight) in sums.items():
        channel_groups[(*key[:-1], key[-1].split("_")[0])].append(total / weight)
    rows = [
        dict(
            zip(
                ("run", "phase", "region", "mode", "lead_days", "variable", "rmse"),
                (*key, np.sqrt(np.mean(values))),
                strict=True,
            )
        )
        for key, values in sorted(channel_groups.items())
    ]
    with (output / "native-controls.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lookup = {
        tuple(
            row[k] for k in ("run", "phase", "region", "mode", "lead_days", "variable")
        ): row["rmse"]
        for row in rows
    }
    fig, axes = plt.subplots(4, 2, figsize=(11, 12), sharex=True, squeeze=False)
    styles = [
        ("A", "inferred_mean", "#286cb0", "o", "A evolved"),
        ("B", "inferred_mean", "#c84638", "s", "B ensemble mean evolved"),
        ("A", "true_interior", "#2a8c54", "^", "True interior evolved"),
        ("A", "true_persistence", "#686868", "D", "True interior persistence"),
        ("A", "inferred_persistence", "#935db0", "v", "A inferred persistence"),
        ("B", "inferred_persistence", "#ce8e23", "P", "B inferred persistence"),
    ]
    for column, phase in enumerate(("om4", "observation")):
        for index, variable in enumerate(VARIABLES):
            ax = axes[index, column]
            for arm, mode, color, marker, label in styles:
                values = np.array(
                    [
                        [
                            lookup[
                                (
                                    f"{arm}-{seed}",
                                    phase,
                                    "scored_latitudes",
                                    mode,
                                    day,
                                    variable,
                                )
                            ]
                            for day in (0, 5, 15, 30)
                        ]
                        for seed in (1729, 1730)
                    ]
                )
                ax.plot(
                    (0, 5, 15, 30),
                    values.mean(0),
                    color=color,
                    marker=marker,
                    label=label,
                )
                ax.fill_between(
                    (0, 5, 15, 30), values.min(0), values.max(0), color=color, alpha=0.1
                )
            ax.set_ylabel(
                f"{variable} RMSE "
                + (
                    "(°C)"
                    if variable == "thetao"
                    else "(practical salinity)"
                    if variable == "so"
                    else "(m/s)"
                )
            )
            ax.grid(alpha=0.2)
            if index == 0:
                ax.set_title(
                    "After OM4 pretraining"
                    if phase == "om4"
                    else "After observation fine-tuning"
                )
            if index == 3:
                ax.set_xlabel("Forecast lead (days)")
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=9
    )
    fig.suptitle(
        "Native OM4 controls: 24 held-out origins, ±60°\nEqual channel weighting; known SST/SSH excluded; shading spans two seeds"
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    save_png(fig, output / "native-controls.png", dpi=135)
    plt.close(fig)


def maps(reports, output):
    for origin in ("2015-01", "2018-01", "2021-01"):
        with (
            np.load(reports / "A-1729" / "members" / (origin + ".npz")) as a,
            np.load(reports / "B-1729" / "members" / (origin + ".npz")) as b,
        ):
            index = a["channel_names"].tolist().index("so_9")
            target_index = a["interior_channel_indices"].tolist().index(index)
            arrays = [
                a["observed_monthly_interior"][0, target_index],
                a["monthly_states"][0, 0, index],
                b["monthly_states"][:, 0, index].mean(0),
                b["monthly_states"][0, 0, index],
            ]
            mask = a["mask"][index].astype(bool)
            wet = np.concatenate([x[mask & np.isfinite(x)] for x in arrays])
            low, high = np.quantile(wet, [0.01, 0.99])
            # Each panel exactly 720x360 screen pixels: 2x2 pixels per native cell.
            fig = plt.figure(figsize=(16, 10), dpi=100)
            for i, (field, title) in enumerate(
                zip(
                    arrays,
                    (
                        "IAP monthly target",
                        "A deterministic",
                        "B ensemble mean (8)",
                        "B member 1",
                    ),
                    strict=True,
                )
            ):
                left = 0.03 + (i % 2) * 0.5
                bottom = 0.54 - (i // 2) * 0.405
                ax = fig.add_axes((left, bottom, 720 / 1600, 360 / 1000))
                im = ax.imshow(
                    np.ma.masked_where(~mask | ~np.isfinite(field), field),
                    origin="lower",
                    vmin=low,
                    vmax=high,
                    cmap="viridis",
                    interpolation="nearest",
                    aspect="equal",
                )
                for latitude in (-60, 60):
                    row = np.interp(latitude, a["lat"], np.arange(180))
                    ax.axhline(float(row), color="white", linewidth=0.8)
                ax.set_title(title)
                ax.set_xticks([])
                ax.set_yticks([])
            cax = fig.add_axes((0.25, 0.085, 0.5, 0.014))
            fig.colorbar(
                im,
                cax=cax,
                orientation="horizontal",
                extend="both",
                label="Practical salinity; shared 1–99% limits, clipped tails",
            )
            fig.text(
                0.5,
                0.01,
                f"{origin} monthly mean, 550 m, seed 1729. White lines: ±60° observation-loss/scoring boundary.",
                ha="center",
                fontsize=10,
            )
            save_png(fig, output / f"monthly-so9-{origin}.png", dpi=100)
            plt.close(fig)


def ranks(summary, output):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), sharey=True)
    for name, color, marker in (("B-1729", "#286cb0", "o"), ("B-1730", "#c84638", "s")):
        p = summary["runs"][name]["probabilistic"]
        for ax, group, channel in zip(
            axes, ("interior", "surface", "surface"), (None, 0, 1), strict=True
        ):
            values = (
                p[group]["scores"]["rank_weights"]
                if channel is None
                else np.asarray(p[group]["profiles"]["rank_weights"])[:, channel]
            )
            ax.plot(range(9), values, color=color, marker=marker, label=name)
            ax.axhline(1 / 9, color="black", linewidth=0.8)
    for ax, title in zip(
        axes,
        ("Monthly observed interior (pooled)", "SST surface bins", "SSH surface bins"),
        strict=True,
    ):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Observation rank among 8 members")
        ax.set_xticks(range(9))
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Wet-area fractional rank mass")
    axes[0].legend()
    fig.tight_layout()
    save_png(fig, output / "rank-histograms.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    native_controls(args.root / "native-controls", args.output)
    maps(args.root / "ab-reports", args.output)
    ranks(json.loads((args.root / "ab-summary/summary.json").read_text()), args.output)


if __name__ == "__main__":
    main()
