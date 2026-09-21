#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Summarize initialization diagnostics and plot true-state/persistence controls."""

import argparse
import io
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from samudra.experiments.surface_adaptation_analysis import read_bytes

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LICENSE = "SPDX-FileCopyrightText: 2026 Samudra Authors\n\nSPDX-License-Identifier: CC-BY-4.0\n"


def frame(path):
    return pd.read_csv(io.BytesIO(read_bytes(path)))


def variable(channel):
    if channel == "thetao_0":
        return "sst"
    if channel.startswith("uo_") or channel.startswith("vo_"):
        return "velocity"
    return channel.split("_")[0]


def save(fig, output, name):
    is_map = name.startswith("initialization_maps_")
    path = output / (name + (".jpg" if is_map else ".png"))
    kwargs = {"pil_kwargs": {"quality": 85, "optimize": True}} if is_map else {}
    fig.savefig(path, dpi=110, bbox_inches="tight", **kwargs)
    plt.close(fig)
    Path(str(path) + ".license").write_text(LICENSE)


def summarize(raw, wave, output):
    output.mkdir(parents=True, exist_ok=True)
    data = frame(raw / "persistence-origins.csv")
    keys = ["model", "region", "origin_index", "lead_days", "channel"]
    assert not data.duplicated(keys).any()
    assert data.origin_index.nunique() == 99
    assert len(data) == 6 * 3 * 99 * 7 * 77
    assert np.isfinite(data.normalized_mse).all()
    assert (
        data.loc[(data.model == "truth") & (data.lead_days == 0), "normalized_mse"] == 0
    ).all()
    data["variable"] = data.channel.map(variable)
    grouped = data.groupby(
        ["model", "region", "lead_days", "variable"], as_index=False
    ).normalized_mse.mean()
    ts = (
        grouped[grouped.variable.isin(["thetao", "so"])]
        .groupby(["model", "region", "lead_days"], as_index=False)
        .normalized_mse.mean()
        .assign(variable="ts")
    )
    grouped = pd.concat([grouped, ts], ignore_index=True)
    grouped["normalized_rmse"] = np.sqrt(grouped.normalized_mse)
    grouped.to_csv(output / "persistence_summary.csv", index=False)
    existing = frame(wave / "analysis/grouped_metrics.csv")
    reference = existing[
        (existing.arm == "E") & (existing["mode"] == "inferred_persistence")
    ]
    matched = grouped[(grouped.model == "E") & (grouped.lead_days > 0)].merge(
        reference, on=["region", "lead_days", "variable"], suffixes=("_new", "_old")
    )
    assert (
        len(matched) == 3 * 6 * 5
    )  # T, S, SSH, SST and combined T/S; velocity merges u/v.
    max_difference = float(
        np.max(np.abs(matched.normalized_rmse_new - matched.normalized_rmse_old))
    )
    np.testing.assert_allclose(
        matched.normalized_rmse_new, matched.normalized_rmse_old, rtol=0.002, atol=1e-6
    )
    temporal = frame(raw / "temporal-decomposition.csv")
    temporal["variable"] = temporal.channel.map(variable)
    tsummary = temporal.groupby(["model", "region", "variable"], as_index=False).mean(
        numeric_only=True
    )
    for column in ["mean_bias_mse", "amplitude_mse", "pattern_mse"]:
        tsummary[column + "_fraction"] = tsummary[column] / tsummary.mse.replace(
            0, np.nan
        )
    for model in ["original", "E"]:
        for var in ["thetao", "so"]:
            actual = tsummary[
                (tsummary.model == model)
                & (tsummary.region == "global")
                & (tsummary.variable == var)
            ].mse.item()
            expected = grouped[
                (grouped.model == model)
                & (grouped.region == "global")
                & (grouped.variable == var)
                & (grouped.lead_days == 0)
            ].normalized_mse.item()
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-9)
    tsummary.to_csv(output / "temporal_summary.csv", index=False)
    scale = frame(raw / "spatial-scales.csv")
    scale["variable"] = scale.channel.map(variable)
    ssummary = scale.groupby(
        ["model", "region", "box_width_cells", "variable"], as_index=False
    ).mean(numeric_only=True)
    ssummary["anomaly_rms_ratio"] = np.sqrt(
        ssummary.prediction_anomaly_energy
        / ssummary.target_anomaly_energy.replace(0, np.nan)
    )
    ssummary["anomaly_correlation"] = ssummary.cross_anomaly_moment / np.sqrt(
        ssummary.prediction_anomaly_energy * ssummary.target_anomaly_energy
    ).replace(0, np.nan)
    ssummary.to_csv(output / "spatial_summary.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, var, title in zip(
        axes,
        ["thetao", "so", "ts"],
        ["Subsurface temperature", "Salinity", "Equal-variable T/S"],
        strict=True,
    ):

        def persist(model):
            return (
                grouped[
                    (grouped.model == model)
                    & (grouped.region == "global")
                    & (grouped.variable == var)
                ]
                .sort_values("lead_days")
                .normalized_rmse.to_numpy()
            )

        def forecast(arm, mode):
            r = existing[
                (existing.arm == arm)
                & (existing["mode"] == mode)
                & (existing.region == "global")
                & (existing.variable == var)
            ].sort_values("lead_days")
            return r.normalized_rmse.to_numpy()

        leads = [0, 5, 10, 15, 20, 25, 30]
        for values, label, color, style in [
            (
                np.r_[persist("E")[0], forecast("E", "inferred")],
                "Inferred + E evolution",
                "#b33a3a",
                "-",
            ),
            (np.r_[0, forecast("E", "true")], "True + E evolution", "#764f9e", "-"),
            (persist("E"), "Inferred, no evolution", "#b33a3a", "--"),
            (persist("truth"), "True, no evolution", "#764f9e", "--"),
            (
                np.r_[0, forecast("A", "true")],
                "True + pretrained evolution",
                "#777777",
                ":",
            ),
        ]:
            ax.plot(leads, values, label=label, color=color, ls=style, marker=".", ms=4)
        ax.set(
            title=title,
            xlabel="Lead (days)",
            ylabel="Normalized RMSE",
            xticks=[0, 10, 20, 30],
        )
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle(
        "Where does forecast error enter? Arm E; 99 held-out OM4 origins\nLead zero measures the latest reconstructed interior; no evolution means holding that state fixed",
        fontsize=11,
    )
    save(fig, output, "initialization_controls")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for ax, var in zip(axes, ["thetao", "so"], strict=True):
        rows = (
            tsummary[(tsummary.region == "global") & (tsummary.variable == var)]
            .set_index("model")
            .loc[["original", "E"]]
        )
        bottom = np.zeros(2)
        for key, label, color in [
            ("mean_bias_mse", "Persistent mean bias", "#d99000"),
            ("amplitude_mse", "Temporal amplitude mismatch", "#3975ad"),
            ("pattern_mse", "Temporal correlation mismatch", "#8757a7"),
        ]:
            vals = rows[key].to_numpy()
            ax.bar(
                ["Original initializer", "Joint-adapted E"],
                vals,
                bottom=bottom,
                label=label,
                color=color,
            )
            bottom += vals
        ax.set(
            title={"thetao": "Subsurface temperature", "so": "Salinity"}[var],
            ylabel="Lead-zero normalized MSE",
        )
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.suptitle(
        "Exact error decomposition at each wet cell over 99 origins\nCorrelation mismatch is not uniquely spatial displacement; mean bias includes persistent spatial errors",
        fontsize=10,
    )
    save(fig, output, "initialization_decomposition")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    for model, style in [("original", "--"), ("E", "-")]:
        for var, color in [("thetao", "#b33a3a"), ("so", "#3975ad")]:
            rows = ssummary[
                (ssummary.model == model)
                & (ssummary.region == "global")
                & (ssummary.variable == var)
            ].sort_values("box_width_cells")
            for ax, column in zip(
                axes, ["anomaly_rms_ratio", "anomaly_correlation"], strict=True
            ):
                ax.plot(
                    range(3),
                    rows[column],
                    label=f"{model}: {var}",
                    color=color,
                    ls=style,
                    marker="o",
                )
    for ax, label in zip(
        axes,
        ["Predicted / true anomaly RMS", "Anomaly pattern correlation"],
        strict=True,
    ):
        ax.set(
            xticks=range(3),
            xticklabels=["All anomalies", "High-pass 3 cells", "High-pass 9 cells"],
            ylabel=label,
        )
        ax.axhline(1, color="gray", lw=0.7)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.suptitle(
        "Is spatial detail missing at initialization?\nMonthly training climatology removed; mask-normalized box filters, not fixed physical wavelengths",
        fontsize=10,
    )
    save(fig, output, "initialization_scales")
    channels = frame(wave / "raw/E/heldout_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 5), constrained_layout=True)
    manifest = json.loads(read_bytes(wave / "raw/E/manifest.json"))
    for ax, var in zip(axes, ["thetao", "so"], strict=True):
        names = [n for n in manifest["channels"] if variable(n) == var]
        depths = [manifest["depths_m"][int(n.split("_")[1])] for n in names]
        for mode, label, color, style in [
            ("initial", "E at initialization", "#b33a3a", "--"),
            ("inferred", "Inferred + E, day 30", "#b33a3a", "-"),
            ("true", "True + E, day 30", "#764f9e", "-"),
            ("persistence", "True, no evolution, day 30", "#764f9e", "--"),
        ]:
            if mode == "initial":
                values = (
                    temporal[(temporal.model == "E") & (temporal.region == "global")]
                    .set_index("channel")
                    .loc[names]
                    .mse.to_numpy()
                    ** 0.5
                )
            elif mode == "persistence":
                values = (
                    data[
                        (data.model == "truth")
                        & (data.region == "global")
                        & (data.lead_days == 30)
                    ]
                    .groupby("channel")
                    .normalized_mse.mean()
                    .loc[names]
                    .to_numpy()
                    ** 0.5
                )
            else:
                values = (
                    channels[
                        (channels["mode"] == mode)
                        & (channels.region == "global")
                        & (channels.lead_days == 30)
                    ]
                    .set_index("channel")
                    .loc[names]
                    .normalized_rmse.to_numpy()
                )
            ax.plot(values, depths, label=label, color=color, ls=style, marker=".")
        ax.set(
            yscale="log",
            ylabel="Depth (m)",
            xlabel="Normalized RMSE",
            title={"thetao": "Subsurface temperature", "so": "Salinity"}[var],
        )
        ax.invert_yaxis()
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.suptitle(
        "Where is initialization error concentrated? Global, 99 origins\nEach depth has its own normalization; not volume- or heat-content-weighted",
        fontsize=10,
    )
    save(fig, output, "initialization_depths")
    metadata = json.loads(read_bytes(raw / "COMPLETE.json"))
    snapshot = np.load(io.BytesIO(read_bytes(raw / "snapshots.npz")))
    index = metadata["snapshot_indices"][1]
    snapshot_date = data.loc[data.origin_index == index, "origin_time"].iloc[0]
    for var in ["thetao", "so"]:
        fig, axes = plt.subplots(2, 4, figsize=(12, 5.4), sharex=True, sharey=True)
        fig.subplots_adjust(
            left=0.05, right=0.89, bottom=0.10, top=0.83, wspace=0.3, hspace=0.48
        )
        for row, channel in enumerate([var + "_5", var + "_9"]):
            climate = snapshot[f"{index}_climatology_{channel}"]
            truth = snapshot[f"{index}_truth_{channel}"] - climate
            original = snapshot[f"{index}_original_{channel}"] - climate
            adapted = snapshot[f"{index}_E_{channel}"] - climate
            limit = np.nanquantile(np.abs(truth), 0.98)
            for ax, field, title in zip(
                axes[row],
                [truth, original, adapted, adapted - truth],
                [
                    "True anomaly",
                    "Original initializer",
                    "Joint-adapted E",
                    "E minus truth",
                ],
                strict=True,
            ):
                im = ax.pcolormesh(
                    snapshot["longitude"],
                    snapshot["latitude"],
                    field,
                    cmap="RdBu_r",
                    vmin=-limit,
                    vmax=limit,
                    rasterized=True,
                    shading="auto",
                )
                ax.set(
                    title=title,
                    xlabel="Longitude",
                    ylabel=f"{snapshot['depths_m'][int(channel.split('_')[1])]:g} m\nLatitude",
                )
            cax = fig.add_axes((0.92, 0.55 if row == 0 else 0.10, 0.015, 0.28))
            fig.colorbar(im, cax=cax, label="Normalized anomaly / error")
        fig.suptitle(
            f"{var}: fixed middle held-out origin {snapshot_date}\nMonthly climatology removed; common limits within depth, clipped at truth 98th absolute percentile",
            fontsize=10,
        )
        save(fig, output, "initialization_maps_" + var)
    audit = {
        "origins": 99,
        "origin_rows": len(data),
        "max_persistence_rmse_difference_from_wave2": max_difference,
        "scope": "Inference only; original and E initializers; snapshots fixed before evaluation. No observational claim, retraining, or held-out model selection.",
    }
    (output / "diagnostic-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    for path in output.iterdir():
        if path.suffix in {".csv", ".json"}:
            Path(str(path) + ".license").write_text(LICENSE)
    print(json.dumps(audit))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument(
        "--wave", type=Path, default=Path("docs/experiments/surface-wave2-results")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.raw, args.wave, args.output)


if __name__ == "__main__":
    main()
