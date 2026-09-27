# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixed-cohort figures for scratch diffusion versus the unchanged baseline."""

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from samudra.experiments.diffusion_ab_figures import ranks, save_png
from samudra.experiments.diffusion_summary import reduce_records

DEPTHS = np.array(
    [
        2.5,
        10,
        22.5,
        40,
        65,
        105,
        165,
        250,
        375,
        550,
        775,
        1050,
        1400,
        1850,
        2400,
        3100,
        4000,
        5000,
        6000,
    ]
)
VARIABLES = ("thetao", "so", "uo", "vo")
STYLES = (
    ("baseline", "#444444", "^", "Unchanged baseline"),
    ("B-1729", "#286cb0", "o", "Scratch 1729"),
    ("B-1730", "#c84638", "s", "Scratch 1730"),
)


def profiles(summary, output):
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharey=True)
    for name, color, marker, label in STYLES:
        run = summary["baseline"] if name == "baseline" else summary["runs"][name]
        p = run["probabilistic"]
        point = p["point_mass_interior"]
        crps = point if name == "baseline" else p["interior"]
        point_values = np.full(28, np.nan)
        point_values[point["supported_channels"]] = np.sqrt(
            point["profiles"]["mean_squared_error"]
        )
        key = "absolute_error" if name == "baseline" else "fair_crps"
        crps_values = np.full(28, np.nan)
        crps_values[crps["supported_channels"]] = crps["profiles"][key]
        for column in range(2):
            selected = slice(column * 14, (column + 1) * 14)
            axes[0, column].plot(
                point_values[selected],
                DEPTHS[:14],
                color=color,
                marker=marker,
                label=label,
            )
            axes[1, column].plot(
                crps_values[selected],
                DEPTHS[:14],
                color=color,
                marker=marker,
                label=label,
            )
    for column, variable in enumerate(("Temperature", "Salinity")):
        axes[0, column].set_title(variable)
        for row, metric in enumerate(
            ("Mean RMSE", "Fair CRPS; baseline point-mass MAE")
        ):
            axes[row, column].set_xlabel(metric + " (fixed standardized units)")
            axes[row, column].grid(alpha=0.2)
    for ax in axes[:, 0]:
        ax.set_ylabel("Depth (m)")
    axes[0, 0].invert_yaxis()
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Monthly observed interior: 96 origins, ±60°; lower is better")
    fig.tight_layout()
    save_png(fig, output / "interior-depth-profiles.png", dpi=135)
    plt.close(fig)


def annual(summary, output):
    """Average per-origin RMSEs; this is not a pooled space-time RMSE."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    days = (5, 15, 30, 90, 180, 365)
    rows: list[dict[str, Any]] = []
    for name, color, marker, label in STYLES:
        run = summary["baseline"] if name == "baseline" else summary["runs"][name]
        years = run["annual"]
        for column, metric in enumerate(("sst_rmse", "adt_rmse")):
            values = [
                np.mean([year["leads"][str(day)][metric] for year in years.values()])
                for day in days
            ]
            axes[0, column].plot(days, values, color=color, marker=marker, label=label)
            rows.extend(
                dict(
                    run=name,
                    metric=metric,
                    lead=day,
                    lead_unit="day",
                    value=float(value),
                )
                for day, value in zip(days, values, strict=True)
            )
        for column, layer in enumerate(("0_700", "700_2000")):
            values = [
                np.mean(
                    [
                        year["monthly_ohc"][month][layer]
                        for year in years.values()
                        for month in year["monthly_ohc"]
                        if int(month[-2:]) == index
                    ]
                )
                / 1e9
                for index in range(1, 13)
            ]
            axes[1, column].plot(
                range(1, 13), values, color=color, marker=marker, label=label
            )
            rows.extend(
                dict(
                    run=name,
                    metric="ohc_" + layer + "_rmse_GJ_m2",
                    lead=month,
                    lead_unit="month",
                    value=float(value),
                )
                for month, value in enumerate(values, 1)
            )
    for column in range(2):
        axes[0, column].set_xlabel("Forecast lead (days)")
        axes[1, column].set_xlabel("Forecast month")
    for ax, label in zip(
        axes.flat,
        (
            "SST RMSE (°C)",
            "Absolute dynamic topography RMSE (m)",
            "0–700 m OHC RMSE (GJ/m²)",
            "700–2000 m OHC RMSE (GJ/m²)",
        ),
        strict=True,
    ):
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Single-initialization annual forecasts: mean RMSE across three origins\nJanuary 2015, 2018, 2021; no future surface observations"
    )
    fig.tight_layout()
    save_png(fig, output / "annual-forecasts.png", dpi=135)
    plt.close(fig)
    with (output / "annual-forecasts.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def annual_bias(summary, baseline, reports, output):
    """Decompose the exact annual surface RMSE into domain bias and centered error."""
    with np.load(baseline / "members/2015-01.npz") as grid:
        lat = grid["lat"]
        domain = grid["mask"][0].astype(bool) & (np.abs(lat[:, None]) <= 60)
        area = np.cos(np.deg2rad(lat))[:, None] * np.ones(domain.shape)
    rows: list[dict[str, Any]] = []
    for name, _, _, _ in STYLES:
        run = summary["baseline"] if name == "baseline" else summary["runs"][name]
        root = baseline if name == "baseline" else reports / name
        for origin, metrics in run["annual"].items():
            with np.load(root / "annual" / (origin + ".npz")) as data:
                predicted, target = data["surface"], data["reference"]
            for day in (5, 15, 30, 90, 180, 365):
                for channel, variable in enumerate(("sst", "adt")):
                    error = (
                        predicted[day // 5 - 1, channel].astype(float)
                        - target[day // 5 - 1, channel]
                    )
                    valid = domain & np.isfinite(error)
                    weights = np.where(valid, area, 0)
                    clean = np.where(valid, error, 0)
                    bias = float((weights * clean).sum() / weights.sum())
                    mse = float((weights * clean**2).sum() / weights.sum())
                    rmse = np.sqrt(mse)
                    np.testing.assert_allclose(
                        rmse, metrics["leads"][str(day)][variable + "_rmse"], rtol=1e-5
                    )
                    rows.append(
                        dict(
                            run=name,
                            origin=origin,
                            lead_days=day,
                            variable=variable,
                            bias=bias,
                            rmse=rmse,
                            centered_rmse=np.sqrt(max(mse - bias**2, 0)),
                            bias_fraction_of_mse=bias**2 / mse,
                        )
                    )
    with (output / "annual-surface-bias.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def velocity(native, output):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for column, phase in enumerate(("om4", "observation")):
        for seed, color, marker in ((1729, "#286cb0", "o"), (1730, "#c84638", "s")):
            data = json.loads(
                (native / f"B-{seed}" / f"{phase}-summary.json").read_text()
            )
            row = next(
                r
                for r in data["results"]
                if r["region"] == "scored_latitudes" and r["lead_days"] == 0
            )
            for index, variable in enumerate(("uo", "vo")):
                select = [
                    i
                    for i, name in enumerate(data["channels"])
                    if name.startswith(variable + "_")
                ]
                for metric, axis in (
                    ("variance", axes[0, column]),
                    ("zonal_increment_mse", axes[1, column]),
                ):
                    model = np.asarray(
                        row["structure"]["members_structure"][metric], dtype=float
                    )[select]
                    truth = np.asarray(
                        row["structure"]["truth_structure"][metric], dtype=float
                    )[select]
                    axis.plot(
                        DEPTHS,
                        np.divide(
                            model,
                            truth,
                            out=np.full_like(model, np.nan),
                            where=truth > 0,
                        ),
                        color=color,
                        marker=marker if variable == "uo" else "x",
                        label=f"{seed} {variable}",
                    )
        axes[0, column].set_title(
            "After OM4 pretraining"
            if phase == "om4"
            else "After observation fine-tuning"
        )
        for row_index in (0, 1):
            ax = axes[row_index, column]
            ax.axhline(1, color="black", linewidth=0.8)
            ax.set_yscale("log")
            ax.grid(alpha=0.2)
        axes[1, column].set_xlabel("Depth (m)")
    axes[0, 0].set_ylabel("Member spatial variance / OM4 variance")
    axes[1, 0].set_ylabel("Member zonal increment MSE / OM4")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Native velocity sample structure at initialization: 24 origins, ±60°\nMoments computed per field before pooling; ratio 1 matches this moment, not full realism"
    )
    fig.tight_layout()
    save_png(fig, output / "native-velocity-structure.png", dpi=135)
    plt.close(fig)


def native_controls(root, output):
    sums: dict[tuple[str, str, str, str, int, str], list[float]] = defaultdict(
        lambda: [0.0, 0.0]
    )
    for run in ("B-1729", "B-1730"):
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
        ("B", "inferred_mean", "#286cb0", "o", "Ensemble mean evolved"),
        ("B", "inferred_member_1", "#c84638", "s", "Member 1 evolved"),
        ("B", "true_interior", "#2a8c54", "^", "True interior evolved"),
        ("B", "true_persistence", "#686868", "D", "True interior persistence"),
        ("B", "inferred_persistence", "#935db0", "v", "Inferred mean persistence"),
        ("B", "zero_velocity", "#ce8e23", "P", "Zero velocity reference"),
    ]
    for column, phase in enumerate(("om4", "observation")):
        for index, variable in enumerate(VARIABLES):
            ax = axes[index, column]
            for arm, mode, color, marker, label in styles:
                if mode == "zero_velocity" and variable not in ("uo", "vo"):
                    continue
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
        *axes[3, 0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=9
    )
    fig.suptitle(
        "Native OM4 controls: 24 held-out origins, ±60°\nEqual channel weighting; known SST/SSH excluded; shading spans two seeds"
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    save_png(fig, output / "native-controls.png", dpi=115)
    plt.close(fig)


def maps(baseline, reports, output, seed):
    for origin in ("2015-01", "2018-01", "2021-01"):
        with (
            np.load(baseline / "members" / (origin + ".npz")) as a,
            np.load(reports / f"B-{seed}" / "members" / (origin + ".npz")) as b,
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
                        "Unchanged deterministic baseline",
                        f"Scratch {seed}: mean of 8",
                        f"Scratch {seed}: member 1",
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
                f"{origin} monthly mean, 550 m, seed {seed}. White lines: ±60° observation-loss/scoring boundary.",
                ha="center",
                fontsize=10,
            )
            save_png(fig, output / f"monthly-so9-{seed}-{origin}.png", dpi=100)
            plt.close(fig)


def sampling(root, output):
    result = {}
    for seed in (1729, 1730):
        directory = root / f"B-{seed}"
        protocol = json.loads((directory / "protocol.json").read_text())
        if json.loads((directory / "COMPLETE.json").read_text()) != protocol:
            raise ValueError("Sampling completion contract differs")
        records = [
            json.loads((directory / (origin + ".json")).read_text())
            for origin in protocol["origins"]
        ]
        if len(records) != 9 or [r["origin"] for r in records] != protocol["origins"]:
            raise ValueError("Incomplete sampling cohort")
        result[f"B-{seed}"] = {
            str(steps): dict(
                seconds_per_origin=float(
                    np.mean(
                        [r["results"][str(steps)]["elapsed_seconds"] for r in records]
                    )
                ),
                **{
                    group: reduce_records(
                        [r["results"][str(steps)] for r in records], group
                    )
                    for group in ("interior", "surface")
                },
            )
            for steps in (8, 16, 32)
        }
    (output / "sampling-summary.json").write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--sampling", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = json.loads(gzip.decompress(args.summary.read_bytes()))
    annual(summary, args.output)
    annual_bias(summary, args.baseline, args.reports, args.output)
    sampling(args.sampling, args.output)
    profiles(summary, args.output)
    ranks(summary, args.output)
    velocity(args.native, args.output)
    native_controls(args.controls, args.output)
    for seed in (1729, 1730):
        maps(args.baseline, args.reports, args.output, seed)


if __name__ == "__main__":
    main()
