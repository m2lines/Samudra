# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Report matched velocity-loss continuations from saved diagnostics; CPU only."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

VARIABLES = {
    "uo": range(19),
    "vo": range(19, 38),
    "thetao": range(38, 57),
    "so": range(57, 76),
    "zos": [76],
}
VELOCITY_VARIABLES = ["uo", "vo"]
LABELS = {
    "control": "CRPS continuation",
    "velocity-mean-mse": "CRPS + velocity mean MSE",
    "velocity-crps-2p2": "Velocity CRPS × 2.2",
    "velocity-mean-mse-corrected": "CRPS + corrected velocity mean MSE",
    "parent": "Parent raw 2k",
}
MARKERS = ["o", "s", "^", "D", "v", "P"]
MONTHS = ["2014-01", "2014-04", "2014-05"]


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initializer_rows(root, weights):
    rows, bins, sources, records = [], [], [], []
    for path in sorted(root.glob("*.json")):
        rec = json.loads(path.read_text())
        if "metrics" not in rec or "speed_bins" not in rec:
            continue
        recorded_weights = rec.get("weights", weights)
        if recorded_weights != weights:
            raise ValueError(f"Weights mismatch in {path}")
        base = dict(
            variant=rec["variant"],
            step=int(rec["step"]),
            weights=weights,
            month=rec["month"],
            checkpoint_sha256=rec["checkpoint_sha256"],
        )
        metrics = rec["metrics"]
        for variable, ids in VARIABLES.items():
            values = {
                key: float(np.mean(np.asarray(value)[list(ids)]))
                for key, value in metrics.items()
            }
            rows.append(dict(**base, variable=variable, **values))
        for b in rec["speed_bins"]:
            bins.append(
                dict(
                    **base,
                    low=b["low"],
                    high=b["high"],
                    **{
                        key: float(np.sum(b[key]))
                        for key in [
                            "area",
                            "squared_error_sum",
                            "predicted_power_sum",
                            "gold_power_sum",
                        ]
                    },
                )
            )
        sources.append(dict(path=str(path), sha256=fingerprint(path)))
        records.append((root, path, rec, weights))
    if not rows:
        raise ValueError(f"No initializer results in {root}")
    return rows, bins, sources, records


def aggregate_initializers(frame):
    keys = ["variant", "step", "weights", "variable"]
    for group, rr in frame.groupby(keys):
        if sorted(rr.month.tolist()) != MONTHS:
            raise ValueError(
                f"Unmatched or duplicate date cohort: {group}: {rr.month.tolist()}"
            )
        if rr.checkpoint_sha256.nunique() != 1:
            raise ValueError(f"Multiple checkpoints in {group}")
    numeric = list(frame.select_dtypes(include=[np.number]).columns.drop("step"))
    out = frame.groupby(keys, as_index=False)[numeric].mean()
    out["mean_rmse"] = np.sqrt(out.mean_mse)
    out["member_rmse"] = np.sqrt(out.member_mse)
    variance = out.member_mse - out.mean_mse
    if (variance < -1e-7).any():
        raise ValueError("Member MSE below mean MSE beyond roundoff")
    out["spread_rms"] = np.sqrt(np.maximum(variance, 0))
    out["dates"] = len(MONTHS)
    return out


def plots(summary, out, parent_step):
    for weights in summary.weights.unique():
        subset = summary[summary.weights == weights]
        fig, axes = plt.subplots(2, 4, figsize=(17, 7), constrained_layout=True)
        for row, variable in enumerate(VELOCITY_VARIABLES):
            for column, (key, label) in enumerate(
                [
                    ("mean_rmse", "Ensemble-mean RMSE"),
                    ("member_rmse", "Individual-member RMSE"),
                    ("fair_crps", "Fair CRPS"),
                    ("spread_rms", "Spread RMS"),
                ]
            ):
                ax = axes[row, column]
                for vi, (variant, rr) in enumerate(
                    subset[subset.variable == variable].groupby("variant")
                ):
                    rr = rr.sort_values("step")
                    ax.plot(
                        rr.step - parent_step,
                        rr[key],
                        marker=MARKERS[vi % len(MARKERS)],
                        label=LABELS.get(str(variant), str(variant)),
                    )
                ax.set(
                    xlabel="Updates after parent",
                    ylabel="m/s",
                    title=f"{variable}: {label}",
                )
                ax.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"Three-date OM4 initialization; {weights} weights")
        for suffix in ["png", "pdf"]:
            fig.savefig(out / f"initializer-curves-{weights}.{suffix}", dpi=150)
        plt.close(fig)
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
        for row, variable in enumerate(VELOCITY_VARIABLES):
            for col, (key, label, units) in enumerate(
                [
                    ("absolute", "Mean member absolute error", "m/s"),
                    ("spread", "Subtracted CRPS spread term", "m/s"),
                    ("coverage", "Reference inside member min/max", "Area fraction"),
                ]
            ):
                ax = axes[row, col]
                for vi, (variant, rr) in enumerate(
                    subset[subset.variable == variable].groupby("variant")
                ):
                    rr = rr.sort_values("step")
                    ax.plot(
                        rr.step - parent_step,
                        rr[key],
                        marker=MARKERS[vi % len(MARKERS)],
                        label=LABELS.get(str(variant), str(variant)),
                    )
                if key == "coverage":
                    ax.axhline(
                        7 / 9,
                        color="black",
                        linewidth=0.8,
                        label="Exchangeable 8-member reference",
                    )
                    ax.set_ylim(0, 1)
                ax.set(
                    xlabel="Updates after parent",
                    ylabel=units,
                    title=f"{variable}: {label}",
                )
                ax.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=8)
        axes[0, 2].legend(fontsize=7)
        fig.suptitle(
            f"Pointwise probabilistic diagnostics; {weights}; spatially dependent cells"
        )
        for suffix in ["png", "pdf"]:
            fig.savefig(out / f"probabilistic-curves-{weights}.{suffix}", dpi=150)
        plt.close(fig)


def maps(records, out, checkpoint_label="final continuation"):
    for weights in sorted({entry[3] for entry in records}):
        group = [entry for entry in records if entry[3] == weights]
        latest = max(entry[2]["step"] for entry in group)
        for month in MONTHS:
            selected = [
                entry
                for entry in group
                if entry[2]["month"] == month
                and (entry[2]["step"] == latest or entry[2]["variant"] == "parent")
            ]
            selected = [
                entry
                for entry in selected
                if entry[1].with_name(entry[1].stem + "-maps.npz").exists()
            ]
            if not selected:
                continue
            selected.sort(key=lambda entry: entry[2]["variant"])
            for name in ["uo_0", "vo_0", "uo_9", "vo_9"]:
                fig, axes = plt.subplots(
                    len(selected),
                    4,
                    figsize=(18, 2.2 * len(selected)),
                    squeeze=False,
                    constrained_layout=True,
                    sharex=True,
                    sharey=True,
                )
                for row, (root, path, rec, _) in enumerate(selected):
                    grid = np.load(root / "grid.npz")
                    channel = list(grid["names"]).index(name)
                    si = list(grid["selection"]).index(channel)
                    z = np.load(path.with_name(path.stem + "-maps.npz"))
                    members, gold = z["members"][:, si], z["gold"][si]
                    if members.shape[0] != 8:
                        raise ValueError("Expected eight members for maps")
                    fields = [gold, members.mean(0), members[0], members.mean(0) - gold]
                    for col, field in enumerate(fields):
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
                        axes[row, col].set_ylim(-65, 65)
                        if row == 0:
                            axes[row, col].set_title(
                                [
                                    "OM4 reference",
                                    "Mean of 8",
                                    "Member 1",
                                    "Mean minus reference",
                                ][col]
                            )
                    axes[row, 0].set_ylabel(
                        LABELS.get(rec["variant"], rec["variant"]), fontsize=8
                    )
                fig.colorbar(
                    mesh,
                    ax=axes,
                    location="bottom",
                    shrink=0.6,
                    label="m/s; colors saturate at ±0.5",
                )
                depth = 2.5 if name.endswith("_0") else 550
                component = (
                    "Eastward velocity"
                    if name.startswith("uo")
                    else "Northward velocity"
                )
                fig.suptitle(
                    f"{month}: {component} at {depth:g} m; {weights}; {checkpoint_label}"
                )
                for suffix in ["png", "pdf"]:
                    fig.savefig(
                        out / f"maps-{weights}-{month}-{name}.{suffix}", dpi=140
                    )
                plt.close(fig)


def surface_spectra(records, out):
    rows = []
    for root, path, rec, weights in records:
        maps_path = path.with_name(path.stem + "-maps.npz")
        if not maps_path.exists():
            continue
        grid = np.load(root / "grid.npz")
        saved = np.load(maps_path)
        iy = np.flatnonzero((grid["lat"] >= 20) & (grid["lat"] <= 40))
        ix = np.flatnonzero((grid["lon"] >= 150) & (grid["lon"] <= 190))
        taper = np.outer(np.hanning(len(iy)), np.hanning(len(ix)))
        k = np.sqrt(
            np.fft.fftfreq(len(iy))[:, None] ** 2
            + np.fft.rfftfreq(len(ix))[None, :] ** 2
        )
        edges = np.linspace(0, 0.71, 15)
        for channel_name in ["uo_0", "vo_0"]:
            channel_index = list(grid["names"]).index(channel_name)
            selection_index = list(grid["selection"]).index(channel_index)
            if not grid["mask"][channel_index][np.ix_(iy, ix)].all():
                raise ValueError("Surface spectral window must be entirely wet")
            members, gold = (
                saved["members"][:, selection_index],
                saved["gold"][selection_index],
            )
            fields = np.concatenate([gold[None], members.mean(0)[None], members])[
                :, iy[:, None], ix
            ]
            fields = (fields - fields.mean((-2, -1), keepdims=True)) * taper
            power = np.abs(np.fft.rfft2(fields)) ** 2 / (taper**2).sum()
            for lower, upper in zip(edges[:-1], edges[1:]):
                select = (k >= lower) & (k < upper)
                if not select.any():
                    continue
                values = power[:, select].mean(1)
                rows.append(
                    dict(
                        variant=rec["variant"],
                        step=rec["step"],
                        weights=weights,
                        month=rec["month"],
                        channel=channel_name,
                        k_low=lower,
                        k_high=upper,
                        gold=values[0],
                        mean8=values[1],
                        member1=values[2],
                        mean_member_power=values[2:].mean(),
                    )
                )
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "surface-member-spectra.csv", index=False)
    for weights in frame.weights.unique():
        selected = frame[frame.weights == weights]
        selected = selected[
            (selected.step == selected.step.max()) | (selected.variant == "parent")
        ]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        for ax, channel, title in zip(
            axes, ["uo_0", "vo_0"], ["Eastward", "Northward"]
        ):
            reference = None
            for vi, (variant, rr) in enumerate(
                selected[selected.channel == channel].groupby("variant")
            ):
                avg = rr.groupby("k_low")[["gold", "mean_member_power"]].mean()
                avg = avg[avg.index > 0]
                if reference is not None and not np.allclose(
                    avg.gold, reference, rtol=1e-10
                ):
                    raise ValueError("Spectral references differ across models")
                reference = avg.gold.to_numpy()
                ax.loglog(
                    avg.index,
                    avg.mean_member_power,
                    marker=MARKERS[vi % len(MARKERS)],
                    label=LABELS.get(str(variant), str(variant)),
                )
            ax.loglog(
                avg.index, reference, color="black", linewidth=2, label="OM4 reference"
            )
            ax.set(
                title=f"{title} velocity at 2.5 m",
                xlabel="Cycles per grid cell",
                ylabel="Tapered squared Fourier amplitude",
            )
            ax.grid(alpha=0.2)
        axes[0].legend(fontsize=8)
        fig.suptitle(f"Pacific window: average individual-member spectra; {weights}")
        for suffix in ["png", "pdf"]:
            fig.savefig(out / f"surface-member-spectra-{weights}.{suffix}", dpi=150)
        plt.close(fig)


def forecasts(root):
    rows, components, sources = [], [], []
    cohort = None
    climatology = None
    for path in sorted(root.glob("*/evaluation/step-*/COMPLETE.json")):
        r = json.loads(path.read_text())
        if "score" not in r:
            continue
        metadata = r["metrics"]
        origins = metadata["origins"]
        if len(origins) != 9 or metadata["members"] != 8:
            raise ValueError(f"Unexpected forecast cohort: {path}")
        if cohort is not None and cohort != origins:
            raise ValueError("Forecast origin mismatch")
        cohort = origins
        current_climatology = r["climatology"]
        if len(r["ratios"]) != 5 or not np.isclose(
            r["score"], np.mean(list(r["ratios"].values())), rtol=1e-10
        ):
            raise ValueError(
                f"Forecast blend differs from the five-component mean: {path}"
            )
        for component, ratio in r["ratios"].items():
            expected_ratio = (
                metadata["metrics"][component]
                / current_climatology["metrics"][component]
            )
            if not np.isclose(ratio, expected_ratio, rtol=1e-10):
                raise ValueError(f"Forecast ratio mismatch for {component}: {path}")
        if climatology is not None and climatology != current_climatology:
            raise ValueError("Forecast climatology mismatch")
        climatology = current_climatology
        variant = path.parents[2].name
        weights = r.get("weights", path.parent.name.rsplit("-", 1)[1])
        row = dict(
            variant=variant,
            step=r["step"],
            weights=weights,
            blend=r["score"],
            checkpoint_sha256=r["checkpoint_sha256"],
        )
        rows.append(row)
        for component, ratio in r["ratios"].items():
            components.append(dict(**row, component=component, ratio=ratio))
        sources.append(dict(path=str(path), sha256=fingerprint(path)))
    return rows, components, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--initializers", type=Path, help="EMA initializer diagnostic directory"
    )
    parser.add_argument("--initializers-raw", type=Path)
    parser.add_argument(
        "--production", type=Path, help="Root containing variant/evaluation directories"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent-step", type=int, default=2000)
    parser.add_argument("--skip-maps", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, bins, sources, records = [], [], [], []
    input_proofs: dict[str, dict[str, object]] = {}
    for root, weights in [(args.initializers, "ema"), (args.initializers_raw, "raw")]:
        if root:
            for month in MONTHS:
                proof_path = root / f"input-{month}.json"
                proof = json.loads(proof_path.read_text())
                if proof.pop("weights", weights) != weights:
                    raise ValueError(
                        f"Initializer proof weights mismatch: {proof_path}"
                    )
                if month in input_proofs and input_proofs[month] != proof:
                    raise ValueError(f"Initializer input mismatch for {month}")
                input_proofs[month] = proof
                sources.append(
                    dict(path=str(proof_path), sha256=fingerprint(proof_path))
                )
            rr, bb, ss, rec = initializer_rows(root, weights)
            rows.extend(rr)
            bins.extend(bb)
            sources.extend(ss)
            records.extend(rec)
    if rows:
        frame = pd.DataFrame(rows)
        frame["mean_rmse"] = np.sqrt(frame.mean_mse)
        frame["member_rmse"] = np.sqrt(frame.member_mse)
        frame["spread_rms"] = np.sqrt(np.maximum(frame.member_mse - frame.mean_mse, 0))
        frame.to_csv(args.output / "initializer-by-date.csv", index=False)
        pairs = []
        pair_keys = ["step", "weights", "month", "variable"]
        for baseline in ["control", "velocity-mean-mse"]:
            reference = frame[frame.variant == baseline]
            if reference.empty:
                continue
            paired = frame[frame.variant != baseline].merge(
                reference,
                on=pair_keys,
                suffixes=("", "_baseline"),
                validate="many_to_one",
            )
            for _, rr in paired.iterrows():
                for metric in [
                    "mean_rmse",
                    "member_rmse",
                    "fair_crps",
                    "spread_rms",
                    "coverage",
                ]:
                    value, base = rr[metric], rr[metric + "_baseline"]
                    pairs.append(
                        dict(
                            variant=rr.variant,
                            baseline=baseline,
                            **{k: rr[k] for k in pair_keys},
                            metric=metric,
                            value=value,
                            baseline_value=base,
                            difference=value - base,
                            relative_change_percent=100 * (value / base - 1)
                            if base != 0
                            else None,
                        )
                    )
        if pairs:
            pd.DataFrame(pairs).to_csv(
                args.output / "initializer-paired-deltas.csv", index=False
            )
        summary = aggregate_initializers(frame)
        summary.to_csv(args.output / "initializer-summary.csv", index=False)
        speed = pd.DataFrame(bins)
        speed.to_csv(args.output / "current-strength-by-date.csv", index=False)
        sums = speed.groupby(
            ["variant", "step", "weights", "low", "high"], as_index=False
        )[["area", "squared_error_sum", "predicted_power_sum", "gold_power_sum"]].sum()
        sums["rmse_vs_zero"] = np.sqrt(sums.squared_error_sum / sums.gold_power_sum)
        sums["power_ratio"] = sums.predicted_power_sum / sums.gold_power_sum
        sums.to_csv(args.output / "current-strength-summary.csv", index=False)
        for weights in sums.weights.unique():
            selected = sums[sums.weights == weights]
            selected = selected[
                (selected.step == selected.step.max()) | (selected.variant == "parent")
            ]
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
            for vi, (variant, rr) in enumerate(selected.groupby("variant")):
                rr = rr.sort_values("low")
                for ax, key in zip(axes, ["rmse_vs_zero", "power_ratio"]):
                    ax.plot(
                        range(len(rr)),
                        rr[key],
                        marker=MARKERS[vi % len(MARKERS)],
                        label=LABELS.get(str(variant), str(variant)),
                    )
            for ax, title in zip(
                axes,
                [
                    "RMSE relative to zero velocity",
                    "Predicted / reference squared velocity",
                ],
            ):
                ax.set_xticks(
                    range(6),
                    ["0–.01", ".01–.03", ".03–.1", ".1–.3", ".3–1", ">1"],
                    rotation=25,
                )
                ax.axhline(1, color="black", linewidth=0.8)
                ax.set(title=title, xlabel="Reference vector speed (m/s)")
                ax.grid(alpha=0.2)
            axes[0].legend(fontsize=8)
            fig.suptitle(f"Current-strength breakdown at final checkpoint; {weights}")
            for suffix in ["png", "pdf"]:
                fig.savefig(
                    args.output / f"current-strength-{weights}.{suffix}", dpi=150
                )
            plt.close(fig)
        plots(summary, args.output, args.parent_step)
        if not args.skip_maps:
            maps(records, args.output)
            surface_spectra(records, args.output)
    if args.production:
        rr, cc, ss = forecasts(args.production)
        if not rr:
            raise ValueError("No completed forecast receipts")
        pd.DataFrame(rr).to_csv(args.output / "forecast-summary.csv", index=False)
        component_frame = pd.DataFrame(cc)
        component_frame.to_csv(args.output / "forecast-components.csv", index=False)
        for weights in component_frame.weights.unique():
            selected = component_frame[component_frame.weights == weights]
            selected = selected[selected.step == selected.step.max()]
            table = selected.pivot(index="component", columns="variant", values="ratio")
            table = table.rename(
                columns=LABELS,
                index={
                    "sst_rmse": "SST",
                    "velocity_rmse": "SSH-derived velocity",
                    "eke_rmse": "EKE",
                    "ohc_0_700_rmse": "OHC 0–700 m",
                    "ohc_700_2000_rmse": "OHC 700–2000 m",
                },
            )
            ax = table.plot.bar(figsize=(11, 5), rot=15)
            ax.axhline(1, color="black", linewidth=0.8)
            ax.set(
                xlabel="",
                ylabel="RMSE / climatology RMSE",
                title=f"Observation forecast components; {weights}",
            )
            ax.grid(axis="y", alpha=0.2)
            ax.figure.tight_layout()
            for suffix in ["png", "pdf"]:
                ax.figure.savefig(
                    args.output / f"forecast-components-{weights}.{suffix}", dpi=150
                )
            plt.close(ax.figure)
        sources.extend(ss)
    if not sources:
        raise ValueError("No results provided")
    (args.output / "provenance.json").write_text(
        json.dumps(
            dict(
                sources=sources,
                script_sha256=fingerprint(Path(__file__)),
                initializer_dates=MONTHS,
                parent_step=args.parent_step,
                aggregation="Equal date/depth MSE then square root; current-strength bins pool weighted cell-depth sums",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
