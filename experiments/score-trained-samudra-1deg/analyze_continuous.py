# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit complete trajectories and summarize member-versus-mean diagnostics."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analyze_score import STYLES, save_csv, save_figure


def grouped_slots(values):
    """Keep nonlinear member statistics separate from the statistic of the mean."""
    return {
        "truth": values[:, 0],
        "mean": values[:, 1],
        "members": values[:, 2:].mean(1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    all_scores, all_global, structure_rows, vertical_rows, phase_rows = (
        [],
        [],
        [],
        [],
        [],
    )
    reference, reference_global = None, None
    for arm in "ABCD":
        root = args.input / arm
        manifest = json.loads((root / "manifest.json").read_text())
        complete = json.loads((root / "COMPLETE.json").read_text())
        assert complete == {"frames": 598, "score_rows": 46046}
        assert (
            not manifest["partial"]
            and manifest["calls"] == 299
            and manifest["checkpoint_epoch"] == 70
        )
        assert (
            manifest["origin"] == "2014-10-15 12:00:00"
            and manifest["last_target"] == "2022-12-24 12:00:00"
        )
        scores = pd.read_csv(root / "scores.csv.gz")
        bulk = pd.read_csv(root / "global.csv.gz")
        assert (
            len(scores) == 46046 and not scores.duplicated(["target", "variable"]).any()
        )
        assert np.isfinite(scores.select_dtypes("number")).all().all()
        assert np.isfinite(bulk.select_dtypes("number")).all().all()
        scores["family"] = scores.variable.str.split("_").str[0]
        scores["year"] = scores.target.str[:4].astype(int)
        all_scores.append(scores)
        all_global.append(bulk)
        save_csv(
            scores.groupby(["target", "nominal_days", "elapsed_days"])
            .mean(numeric_only=True)
            .reset_index(),
            args.output / f"continuous-scores-{arm}.csv",
        )
        save_csv(bulk, args.output / f"continuous-global-{arm}.csv.gz")
        structure = np.load(root / "structure.npz")
        keys = [k for k in structure.files if k not in ("times", "variables", "slots")]
        for key in keys:
            assert np.isfinite(structure[key]).all()
        current_truth = {k: structure[k][:, 0] for k in keys}
        if reference is None:
            reference = current_truth
            reference_global = (
                bulk[bulk.slot == 0].drop(columns="arm").reset_index(drop=True)
            )
        else:
            assert reference_global is not None
            for key, value in current_truth.items():
                np.testing.assert_allclose(value, reference[key], rtol=1e-6, atol=1e-7)
            pd.testing.assert_frame_equal(
                bulk[bulk.slot == 0].drop(columns="arm").reset_index(drop=True),
                reference_global,
            )
        assert len(structure["times"]) == 598
        assert structure["variables"].tolist() == manifest["variables"]
        for key in (
            "spatial_std",
            "zonal_increment_rms_1",
            "zonal_increment_rms_4",
            "zonal_increment_rms_16",
            "temporal_increment_rms",
        ):
            groups = grouped_slots(structure[key])
            truth = groups["truth"].mean(0)
            for slot, values in groups.items():
                for channel, name in enumerate(manifest["variables"]):
                    physical = values[:, channel].mean()
                    structure_rows.append(
                        {
                            "arm": arm,
                            "statistic": key,
                            "slot": slot,
                            "variable": name,
                            "physical_value": physical,
                            "truth_value": truth[channel],
                            "ratio_to_truth": physical / truth[channel],
                        }
                    )
                if key == "temporal_increment_rms":
                    for parity, label in ((0, "across_calls"), (1, "within_call")):
                        phase = values[parity::2].mean(0)
                        true_phase = groups["truth"][parity::2].mean(0)
                        for channel, name in enumerate(manifest["variables"]):
                            phase_rows.append(
                                {
                                    "arm": arm,
                                    "slot": slot,
                                    "phase": label,
                                    "variable": name,
                                    "physical_value": phase[channel],
                                    "truth_value": true_phase[channel],
                                    "ratio_to_truth": phase[channel]
                                    / true_phase[channel],
                                }
                            )
        for key in (
            "uo_adjacent_depth_rms",
            "vo_adjacent_depth_rms",
            "thetao_adjacent_depth_rms",
            "so_adjacent_depth_rms",
            "thetao_so_covariance",
            "uo_vo_covariance",
        ):
            for slot, values in grouped_slots(structure[key]).items():
                for depth, value in enumerate(values.mean(0)):
                    vertical_rows.append(
                        {
                            "arm": arm,
                            "statistic": key,
                            "slot": slot,
                            "depth_index": depth,
                            "value": value,
                        }
                    )
        # Every saved field remains finite, dry-zero, and mean/member consistent.
        for path in root.glob("fields-*.npz"):
            fields = np.load(path)
            assert np.isfinite(fields["physical"]).all()
            assert not fields["physical"][..., ~fields["mask"].astype(bool)].any()
            np.testing.assert_allclose(
                fields["physical"][1],
                fields["physical"][2:].mean(0),
                rtol=1e-5,
                atol=1e-5,
            )
    scores = pd.concat(all_scores)
    bulk = pd.concat(all_global)
    structure_df, vertical_df, phase_df = (
        pd.DataFrame(structure_rows),
        pd.DataFrame(vertical_rows),
        pd.DataFrame(phase_rows),
    )
    for arm in "ABCD":
        save_csv(
            structure_df[structure_df.arm == arm],
            args.output / f"continuous-structure-{arm}.csv",
        )
        save_csv(
            phase_df[phase_df.arm == arm], args.output / f"continuous-phase-{arm}.csv"
        )
    save_csv(vertical_df, args.output / "continuous-vertical.csv")
    save_csv(
        scores.groupby(["arm", "year", "family"]).mean(numeric_only=True).reset_index(),
        args.output / "continuous-year-family.csv",
    )
    yearly = scores.groupby(["arm", "year"]).mean(numeric_only=True).reset_index()
    save_csv(yearly, args.output / "continuous-year-summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    for arm, (color, marker) in STYLES.items():
        monthly = (
            scores[scores.arm == arm]
            .assign(month=lambda d: d.target.str[:7])
            .groupby("month")
            .mean(numeric_only=True)
        )
        x = pd.to_datetime(monthly.index)
        for ax, metric, title in zip(
            axes,
            ("rmse", "crps_empirical"),
            ("Ensemble-mean RMSE", "Empirical CRPS (C/D: 4 members)"),
            strict=True,
        ):
            ax.plot(
                x, monthly[metric], color=color, marker=marker, markevery=8, label=arm
            )
            ax.set(
                title=title,
                ylabel="Mean normalized channel score",
                xlabel="Valid date (no resets)",
            )
            ax.grid(alpha=0.2)
            ax.legend()
    save_figure(fig, args.output / "continuous-scores.jpg")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
    bulk_rows = []
    for arm, (color, marker) in STYLES.items():
        d = bulk[bulk.arm == arm]
        truth = d[d.slot == 0].set_index("target")
        for slot, values in (
            ("mean", d[d.slot == 1].set_index("target")),
            ("members", d[d.slot >= 2].groupby("target").mean(numeric_only=True)),
        ):
            values = values[["volume_mean_thetao", "volume_mean_so", "volume_mean_ke"]]
            delta = values - truth[values.columns]
            monthly = (
                delta.assign(month=lambda x: x.index.str[:7]).groupby("month").mean()
            )
            x = pd.to_datetime(monthly.index)
            if slot == "mean":
                for ax, metric in zip(
                    axes[:2], ["volume_mean_thetao", "volume_mean_so"], strict=True
                ):
                    ax.plot(
                        x,
                        monthly[metric],
                        color=color,
                        marker=marker,
                        markevery=8,
                        label=arm,
                    )
            if slot == "mean" or arm in "CD":
                ratio = (values.volume_mean_ke / truth.volume_mean_ke).to_frame("ratio")
                ratio["month"] = ratio.index.str[:7]
                ratio = ratio.groupby("month").mean()
                axes[2].plot(
                    pd.to_datetime(ratio.index),
                    ratio.ratio,
                    color=color,
                    marker=marker if slot == "mean" else ("P" if arm == "C" else "X"),
                    markerfacecolor="white"
                    if slot == "mean" and arm in "CD"
                    else color,
                    markevery=8,
                    label=f"{arm} {slot}" if arm in "CD" else arm,
                )
            final = values.index >= "2022"
            bulk_rows.append(
                {
                    "arm": arm,
                    "slot": slot,
                    "thetao_bias_2022": delta.loc[final, "volume_mean_thetao"].mean(),
                    "so_bias_2022": delta.loc[final, "volume_mean_so"].mean(),
                    "ke_ratio_2022": values.loc[final, "volume_mean_ke"].mean()
                    / truth.loc[final, "volume_mean_ke"].mean(),
                }
            )
    for ax, title in zip(
        axes,
        [
            "Depth-weighted temperature bias (°C)",
            "Depth-weighted salinity bias",
            "Depth-weighted KE / OM4 KE",
        ],
        strict=True,
    ):
        ax.set(title=title, xlabel="Valid date")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[2].axhline(1, color="black", alpha=0.4)
    save_figure(fig, args.output / "continuous-bulk.jpg")
    save_csv(pd.DataFrame(bulk_rows), args.output / "continuous-bulk-summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    stats = [
        "spatial_std",
        "zonal_increment_rms_1",
        "zonal_increment_rms_4",
        "zonal_increment_rms_16",
        "temporal_increment_rms",
    ]
    labels = ["Spatial SD", "Zonal Δ1", "Zonal Δ4", "Zonal Δ16", "Time Δ1"]
    series = [
        ("A", "mean"),
        ("B", "mean"),
        ("C", "mean"),
        ("C", "members"),
        ("D", "mean"),
        ("D", "members"),
    ]
    for ax, variable in zip(
        axes.flat, ["zos", "thetao_0", "so_9", "uo_9"], strict=True
    ):
        for index, (arm, slot) in enumerate(series):
            d = structure_df[
                (structure_df.arm == arm)
                & (structure_df.slot == slot)
                & (structure_df.variable == variable)
            ].set_index("statistic")
            color, _ = STYLES[arm]
            ax.bar(
                np.arange(5) + (index - 2.5) * 0.125,
                d.loc[stats, "ratio_to_truth"],
                width=0.12,
                color=color,
                alpha=0.45 if slot == "mean" and arm in "CD" else 1,
                hatch="//" if slot == "mean" and arm in "CD" else None,
                label=f"{arm} {slot}" if arm in "CD" else arm,
            )
        ax.set(
            title=variable,
            ylabel="Mean diagnostic / OM4 diagnostic",
            xticks=np.arange(5),
            xticklabels=labels,
        )
        ax.axhline(1, color="black", lw=1)
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(ncol=3, fontsize=8)
    save_figure(fig, args.output / "continuous-structure.jpg")
    print(yearly[yearly.year == 2022].to_string(index=False))
    print(pd.DataFrame(bulk_rows).to_string(index=False))
    print(
        phase_df[(phase_df.variable == "so_9") & (phase_df.slot != "truth")].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
