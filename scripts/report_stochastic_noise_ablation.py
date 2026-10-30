# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Matched noise-ablation curves from saved initialization/forecast outputs; CPU."""

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
from report_stochastic_velocity_wave2 import maps as initializer_maps

VARIANTS = ["B-control", "B-multiscale-noise", "B-noise-processing"]
LABELS = ["Unchanged noise", "Multiscale noise", "Learned noise processing"]
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
        if status != dict(
            step=2000, counts=dict(om4=1000, observation=1000), complete=True
        ):
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


def paired_differences(frame):
    keys = ["step", "weights", "month", "variable"]
    control = frame[frame.variant == "B-control"]
    matched = frame[frame.variant != "B-control"].merge(
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
                    VARIANTS, LABELS, ["o", "s", "^"], strict=True
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
            handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=4
        )
        fig.suptitle(
            f"Three-date OM4 initialization, {weights.upper()}; four members in training, eight at evaluation",
            y=1.055,
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
    if len(forecast) != 6:
        raise ValueError("Expected exactly three arms × raw/EMA final forecasts")
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
    """Reuse the verified map renderer for final January EMA outputs only."""
    selected: list[Any] = []
    for variant in VARIANTS:
        _, _, _, records = initializer_rows(root / variant / "initializer-ema", "ema")
        selected.extend(
            record
            for record in records
            if record[2]["step"] == 2000 and record[2]["month"] == "2014-01"
        )
    if len(selected) != len(VARIANTS):
        raise ValueError("Expected one final January EMA map case per variant")
    output.mkdir(parents=True, exist_ok=True)
    initializer_maps(
        selected, output, checkpoint_label="2,000 total updates from scratch"
    )
    files = sorted(output.glob("maps-ema-2014-01-*.*"))
    files = [path for path in files if path.suffix in (".png", ".pdf")]
    if len(files) != 8:
        raise ValueError("Expected four January EMA map channels in PNG and PDF")
    for path in files:
        Path(str(path) + ".license").write_text(
            "SPDX-FileCopyrightText: 2026 Samudra Authors\nSPDX-License-Identifier: CC-BY-4.0\n"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--maps-output",
        type=Path,
        help="Optionally render final January EMA u/v maps at 2.5 and 550 m",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame, sources, proofs, checkpoints = load_inputs(args.production)
    for column in ["mean", "member"]:
        frame[f"{column}_rmse"] = np.sqrt(frame[f"{column}_mse"])
    frame["spread_rms"] = np.sqrt(np.maximum(frame.member_mse - frame.mean_mse, 0))
    summary = aggregate_initializers(frame)
    differences = paired_differences(frame)
    forecast, components, fsources = verify_forecasts(args.production, checkpoints)
    sources.extend(fsources)
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
