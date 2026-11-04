# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Shared-input initializer capacity report from matched saved outputs; CPU."""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from report_stochastic_velocity_wave2 import (  # type: ignore[import-not-found]
    MONTHS,
    aggregate_initializers,
    fingerprint,
    forecasts,
    initializer_rows,
)

VARIANTS = [
    "B-learned-control",
    "B-shared",
    "B-shared-deep",
    "B-shared-wide",
    "B-shared-deep-wide",
]
LABELS = [
    "Learned-noise control",
    "Shared baseline",
    "Shared deep",
    "Shared wide",
    "Shared deep + wide",
]
STEPS = [500, 1000, 2000]
UNITS = dict(uo="m/s", vo="m/s", thetao="°C", so="salinity units", zos="m")
METRICS = ["mean_rmse", "member_rmse", "fair_crps", "spread_rms", "coverage"]
TITLES = ["Mean RMSE", "Member RMSE", "Fair CRPS", "Spread RMS", "Min–max coverage"]


def matching_arrays(left, right, keys):
    for key in keys:
        np.testing.assert_array_equal(left[key], right[key], err_msg=key)


def load_inputs(root):
    rows, sources = [], []
    proofs: dict[str, Any] = {}
    checkpoints: dict[tuple[str, int], str] = {}
    common_grid = None
    reference_maps: dict[str, np.ndarray] = {}
    for variant in VARIANTS:
        production = root / variant
        status = json.loads((production / "COMPLETE.json").read_text())
        expected = dict(
            step=2000, counts=dict(om4=1000, observation=1000), complete=True
        )
        if status != expected:
            raise ValueError(f"Unexpected training completion: {variant}")
        for weights in ["raw", "ema"]:
            folder = production / f"initializer-{weights}"
            done = json.loads((folder / "COMPLETE.json").read_text())
            if not done["complete"] or len(done["results"]) != 9:
                raise ValueError("Initializer cohort incomplete")
            grid = np.load(folder / "grid.npz")
            arrays = {key: grid[key] for key in grid.files}
            if common_grid is None:
                common_grid = arrays
            matching_arrays(
                arrays,
                common_grid,
                ["lat", "lon", "mask", "names", "selection"],
            )
            sources.append(
                dict(
                    path=str(folder / "grid.npz"),
                    sha256=fingerprint(folder / "grid.npz"),
                )
            )
            for month in MONTHS:
                path = folder / f"input-{month}.json"
                proof = json.loads(path.read_text())
                if proof.pop("weights") != weights or proof["members"] != 8:
                    raise ValueError("Input weights/member mismatch")
                if month in proofs and proofs[month] != proof:
                    raise ValueError(f"Input cohort differs: {path}")
                proofs[month] = proof
                sources.append(dict(path=str(path), sha256=fingerprint(path)))
            rr, _, ss, records = initializer_rows(folder, weights)
            if len(records) != 9 or sorted({r[2]["step"] for r in records}) != STEPS:
                raise ValueError("Unexpected initializer steps/count")
            for _, path, rec, _ in records:
                if rec["variant"] != variant:
                    raise ValueError("Mislabelled initializer variant")
                key = (variant, rec["step"])
                if key in checkpoints and checkpoints[key] != rec["checkpoint_sha256"]:
                    raise ValueError(
                        "Raw and EMA or dates have different checkpoint hashes"
                    )
                checkpoints[key] = rec["checkpoint_sha256"]
                if rec["step"] == 2000:
                    maps_path = path.with_name(path.stem + "-maps.npz")
                    with np.load(maps_path) as maps:
                        if rec["month"] in reference_maps:
                            np.testing.assert_array_equal(
                                maps["gold"], reference_maps[rec["month"]]
                            )
                        else:
                            reference_maps[rec["month"]] = maps["gold"].copy()
                    sources.append(
                        dict(path=str(maps_path), sha256=fingerprint(maps_path))
                    )
            rows.extend(rr)
            sources.extend(ss)
    return pd.DataFrame(rows), sources, proofs, checkpoints


def channel_metrics(root):
    rows = []
    for variant in VARIANTS:
        for weights in ("raw", "ema"):
            folder = root / variant / f"initializer-{weights}"
            with np.load(folder / "grid.npz") as grid:
                names = grid["names"].tolist()
            _, _, _, records = initializer_rows(folder, weights)
            for _, _, rec, _ in records:
                for channel, name in enumerate(names):
                    row = dict(
                        variant=variant,
                        weights=weights,
                        step=rec["step"],
                        month=rec["month"],
                        channel=name,
                    )
                    row.update(
                        {key: values[channel] for key, values in rec["metrics"].items()}
                    )
                    rows.append(row)
    frame = pd.DataFrame(rows)
    keys = ["variant", "weights", "step", "channel"]
    numeric = list(frame.select_dtypes(include=[np.number]).columns.drop("step"))
    summary = frame.groupby(keys, as_index=False)[numeric].mean()
    for value in (frame, summary):
        value["mean_rmse"] = np.sqrt(value.mean_mse)
        value["member_rmse"] = np.sqrt(value.member_mse)
        value["spread_rms"] = np.sqrt(np.maximum(value.member_mse - value.mean_mse, 0))
    return frame, summary


def paired_differences(frame):
    keys = ["step", "weights", "month", "variable"]
    control = frame[frame.variant == "B-learned-control"]
    matched = frame[frame.variant != "B-learned-control"].merge(
        control, on=keys, suffixes=("", "_control"), validate="many_to_one"
    )
    rows = []
    for _, r in matched.iterrows():
        for metric in METRICS:
            value, baseline = r[metric], r[metric + "_control"]
            rows.append(
                dict(
                    variant=r.variant,
                    **{k: r[k] for k in keys},
                    metric=metric,
                    value=value,
                    control=baseline,
                    difference=value - baseline,
                    relative_change_percent=100 * (value / baseline - 1)
                    if baseline
                    else np.nan,
                )
            )
    return pd.DataFrame(rows)


def draw_initializers(summary, output):
    for weights in ["raw", "ema"]:
        fig, axes = plt.subplots(5, 5, figsize=(19, 15), constrained_layout=True)
        for row, (variable, units) in enumerate(UNITS.items()):
            for col, (metric, title) in enumerate(zip(METRICS, TITLES, strict=True)):
                ax = axes[row, col]
                for variant, label, marker in zip(
                    VARIANTS,
                    LABELS,
                    ["o", "s", "^", "D", "P"][: len(VARIANTS)],
                    strict=True,
                ):
                    rr = summary[
                        (summary.variant == variant)
                        & (summary.weights == weights)
                        & (summary.variable == variable)
                    ].sort_values("step")
                    ax.plot(
                        rr.step,
                        rr[metric] * (100 if metric == "coverage" else 1),
                        marker=marker,
                        label=label,
                    )
                if metric == "coverage":
                    ax.axhline(700 / 9, color="black", linestyle=":", linewidth=0.8)
                    ax.set_ylim(0, 100)
                ax.set(
                    title=f"{variable}: {title}",
                    ylabel="%" if metric == "coverage" else units,
                    xlabel="Total optimizer updates",
                    xticks=STEPS,
                )
                ax.grid(alpha=0.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        handles.append(Line2D([], [], color="black", linestyle=":"))
        labels.append("Calibrated 8-member range: 77.8%")
        fig.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.055), ncol=3
        )
        fig.suptitle(
            f"Three-date OM4 initialization, {weights.upper()}; four members in training, eight at evaluation",
            y=1.095,
        )
        for ext in ["png", "pdf"]:
            fig.savefig(
                output / f"initializer-curves-{weights}.{ext}",
                dpi=150,
                bbox_inches="tight",
            )
        plt.close(fig)


def verify_forecasts(root, checkpoints):
    forecast, components, sources = forecasts(root)
    if len(forecast) != 2 * len(VARIANTS):
        raise ValueError("Expected every configured arm × raw/EMA final forecasts")
    known = set()
    references: dict[str, dict[str, np.ndarray]] = {}
    for row in forecast:
        variant, weights = row["variant"], row["weights"]
        if (
            variant not in VARIANTS
            or weights not in ["raw", "ema"]
            or row["step"] != 2000
            or (variant, weights) in known
        ):
            raise ValueError("Unexpected final forecast")
        known.add((variant, weights))
        if row["checkpoint_sha256"] != checkpoints[(variant, 2000)]:
            raise ValueError("Forecast/initializer checkpoint differs")
        folder = root / variant / f"evaluation/step-02000-{weights}"
        receipt = json.loads((folder / "COMPLETE.json").read_text())
        for month in receipt["metrics"]["origins"]:
            path = folder / f"members-{month}.npz"
            with np.load(path) as z:
                keys = ["channel_names", "mean", "std", "mask", "day30_reference"]
                if month in references:
                    matching_arrays(z, references[month], keys)
                else:
                    references[month] = {key: z[key].copy() for key in keys}
                if z["day30"].shape[0] != 8:
                    raise ValueError("Unexpected forecast ensemble size")
            sources.append(dict(path=str(path), sha256=fingerprint(path)))
    return pd.DataFrame(forecast), pd.DataFrame(components), sources


def draw_forecasts(frame, output):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, weights in zip(axes, ["raw", "ema"], strict=True):
        table = (
            frame[frame.weights == weights]
            .pivot(index="component", columns="variant", values="ratio")
            .reindex(columns=VARIANTS)
        )
        table.columns = LABELS
        table.plot.bar(ax=ax, rot=25)
        ax.axhline(1, color="black", linewidth=0.8)
        ax.set(
            title=f"Final 2k forecast: {weights.upper()}",
            ylabel="Error / climatology error",
            xlabel="Component",
        )
        ax.legend(fontsize=8)
    for ext in ["png", "pdf"]:
        fig.savefig(output / f"forecast-components.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def draw_january_ema_maps(root, output):
    """Final January EMA fields with common limits and full latitude coverage."""
    selected = []
    for variant, label in zip(VARIANTS, LABELS, strict=True):
        folder = root / variant / "initializer-ema"
        _, _, _, records = initializer_rows(folder, "ema")
        record = [
            rec
            for rec in records
            if rec[2]["step"] == 2000 and rec[2]["month"] == "2014-01"
        ]
        if len(record) != 1:
            raise ValueError("Expected one final January EMA map case per variant")
        _, path, _, _ = record[0]
        with np.load(folder / "grid.npz") as z:
            grid = {key: z[key].copy() for key in z.files}
        with np.load(path.with_name(path.stem + "-maps.npz")) as z:
            fields = {key: z[key].copy() for key in z.files}
        selected.append((label, grid, fields))
    output.mkdir(parents=True, exist_ok=True)
    for name in ("uo_0", "vo_0", "uo_9", "vo_9"):
        fig, axes = plt.subplots(
            len(VARIANTS),
            4,
            figsize=(18, 12),
            constrained_layout=True,
            sharex=True,
            sharey=True,
        )
        for row, (label, grid, fields) in enumerate(selected):
            channel = list(grid["names"]).index(name)
            index = list(grid["selection"]).index(channel)
            members, gold = fields["members"][:, index], fields["gold"][index]
            panels = (gold, members.mean(0), members[0], members.mean(0) - gold)
            for col, field in enumerate(panels):
                mesh = axes[row, col].pcolormesh(
                    grid["lon"],
                    grid["lat"],
                    np.where(grid["mask"][channel], field, np.nan),
                    shading="nearest",
                    cmap="RdBu_r",
                    vmin=-0.5,
                    vmax=0.5,
                    rasterized=True,
                )
                axes[row, col].set_ylim(-90, 90)
                axes[row, col].set_aspect("equal")
                if row == 0:
                    axes[row, col].set_title(
                        (
                            "OM4 reference",
                            "Mean of 8",
                            "Member 1",
                            "Mean minus reference",
                        )[col]
                    )
            axes[row, 0].set_ylabel(label, fontsize=9)
        fig.colorbar(
            mesh,
            ax=axes,
            location="bottom",
            shrink=0.6,
            label="m/s; all panels share colors saturating at ±0.5",
        )
        depth = 2.5 if name.endswith("_0") else 550
        component = "Eastward" if name.startswith("uo") else "Northward"
        fig.suptitle(
            f"January 2014: {component} velocity at {depth:g} m; EMA at 2,000 mixed updates"
        )
        for suffix in ("png", "pdf"):
            path = output / f"maps-ema-2014-01-{name}.{suffix}"
            fig.savefig(path, dpi=140)
            Path(str(path) + ".license").write_text(
                "SPDX-FileCopyrightText: 2026 Samudra Authors\nSPDX-License-Identifier: CC-BY-4.0\n"
            )
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--maps-output",
        type=Path,
        help="Optionally render final January EMA u/v maps at 2.5 and 550 m",
    )
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    frame, sources, proofs, checkpoints = load_inputs(args.production)
    for column in ["mean", "member"]:
        frame[f"{column}_rmse"] = np.sqrt(frame[f"{column}_mse"])
    frame["spread_rms"] = np.sqrt(np.maximum(frame.member_mse - frame.mean_mse, 0))
    summary = aggregate_initializers(frame)
    differences = paired_differences(frame)
    forecast, components, fsources = verify_forecasts(args.production, checkpoints)
    sources.extend(fsources)
    channels, channel_summary = channel_metrics(args.production)
    channels.to_csv(args.output / "initializer-channel-by-date.csv", index=False)
    channel_summary.to_csv(args.output / "initializer-channel-summary.csv", index=False)
    frame.to_csv(args.output / "initializer-by-date.csv", index=False)
    summary.to_csv(args.output / "initializer-summary.csv", index=False)
    differences.to_csv(args.output / "initializer-paired-differences.csv", index=False)
    consistency = (
        differences[differences.step == 2000]
        .groupby(["variant", "weights", "variable", "metric"])
        .agg(
            dates=("difference", "size"),
            lower_on_dates=("difference", lambda x: int((x < 0).sum())),
            higher_on_dates=("difference", lambda x: int((x > 0).sum())),
            mean_difference=("difference", "mean"),
            minimum_relative_change_percent=("relative_change_percent", "min"),
            maximum_relative_change_percent=("relative_change_percent", "max"),
        )
        .reset_index()
    )
    consistency.to_csv(args.output / "final-per-date-consistency.csv", index=False)
    forecast.to_csv(args.output / "forecast-summary.csv", index=False)
    components.to_csv(args.output / "forecast-components.csv", index=False)
    draw_initializers(summary, args.output)
    draw_forecasts(components, args.output)
    if args.maps_output is not None:
        draw_january_ema_maps(args.production, args.maps_output)
    (args.output / "COMPLETE.json").write_text(
        json.dumps(
            dict(
                complete=True,
                variants=VARIANTS,
                steps=STEPS,
                weights=["raw", "ema"],
                initializer_dates=MONTHS,
                forecast_origins=9,
                sources=sources,
                input_proofs=proofs,
                script_sha256=fingerprint(Path(__file__)),
                aggregation="Cosine-weighted wet spatial MSE per channel, equal mean over variable depths and three dates, then square root. CRPS and coverage equal mean over depths and dates. Forecast blend unchanged five-component climatology-relative mean.",
                checks="Initializer masks/normalization/input receipts/gold maps match; final checkpoint fingerprints match raw/EMA and forecasts; forecast masks/scales/channel names/day30 references match across arms and weights; nine origins and climatology receipts match.",
            ),
            indent=2,
        )
        + "\n"
    )
    for path in args.output.iterdir():
        if path.is_file() and not path.name.endswith(".license"):
            Path(str(path) + ".license").write_text(
                "SPDX-FileCopyrightText: 2026 Samudra Authors\nSPDX-License-Identifier: CC-BY-4.0\n"
            )


if __name__ == "__main__":
    main()
