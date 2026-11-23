# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Plot saved EMA validation histories; no training or inference is performed."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"mixed": "#d65f00", "obs-only": "#006eb8"}
LABELS = {"mixed": "LLC + OBS", "obs-only": "OBS only"}


def histories(root):
    result = []
    for arm, files in [
        (
            "mixed",
            [
                "mixed-observation-validation.csv",
                "mixed-long-observation-validation.csv",
            ],
        ),
        ("obs-only", ["obs-only-validation.csv", "obs-only-long-validation.csv"]),
    ]:
        data = pd.concat([pd.read_csv(root / f) for f in files], ignore_index=True)
        assert len(data) == 110 * 6
        assert not data.duplicated(["updates", "lead_hours", "variable"]).any()
        assert set(data.checkpoint_weights) == {"EMA"}
        assert set(data.updates) == set(range(100, 11001, 100))
        assert set(zip(data.lead_hours, data.variable)) == {
            (h, v) for h in (24, 48, 72) for v in ("thetao_0", "zos")
        }
        assert np.isfinite(data.model_rmse).all() and (data.model_rmse >= 0).all()
        # The official aggregate pools MSE across origins before square roots,
        # then averages the six normalized variable/lead RMSEs equally.
        reconstructed = data.groupby("updates").model_normalized_mse.apply(
            lambda x: np.sqrt(x).mean()
        )
        recorded = data.groupby("updates").aggregate_normalized_rmse.first()
        np.testing.assert_allclose(reconstructed, recorded, rtol=1e-12)
        data["arm"] = arm
        result.append(data)
    return pd.concat(result, ignore_index=True)


def shade(ax):
    ax.axvspan(10000, 11000, color="#dbe8f1", alpha=0.65, zorder=0)
    ax.axvline(10000, color="#768899", linewidth=1, linestyle=":")
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Total optimizer updates")


def paired_summary(origin_csv, aggregate, output):
    records = pd.read_csv(origin_csv)
    assert len(records) == 192
    assert not records.duplicated(
        ["arm", "updates", "origin", "lead_hours", "variable"]
    ).any()
    assert (
        np.isfinite(records.mse_normalized).all()
        and (records.mse_normalized >= 0).all()
    )
    rng = np.random.default_rng(20261121)
    draws = rng.integers(0, 8, size=(20000, 8))
    rows, summaries = [], {}

    def score(values):
        return np.sqrt(values.mean(axis=-3)).mean(axis=(-1, -2))

    for update in (10000, 11000):
        arrays, origins = [], []
        for arm in ("mixed", "obs-only"):
            cases = records[(records.arm == arm) & (records.updates == update)]
            origin_names = sorted(cases.origin.unique())
            assert len(origin_names) == 8
            origins.append(origin_names)
            expected_index = pd.MultiIndex.from_product(
                [origin_names, (24, 48, 72), ("thetao_0", "zos")],
                names=["origin", "lead_hours", "variable"],
            )
            aligned = cases.set_index(["origin", "lead_hours", "variable"]).reindex(
                expected_index
            )
            assert not aligned.mse_normalized.isna().any()
            values = aligned.mse_normalized.to_numpy().reshape(8, 3, 2)
            assert values.shape == (8, 3, 2)
            arrays.append(values)
            expected = aggregate.loc[
                (aggregate.arm == arm) & (aggregate.updates == update),
                "aggregate_normalized_rmse",
            ].item()
            np.testing.assert_allclose(score(values), expected, rtol=2e-4, atol=2e-7)
        assert origins[0] == origins[1]
        mixed, obs = arrays
        bootstrap = 100 * (score(mixed[draws]) / score(obs[draws]) - 1)
        leave_one_out = [
            100
            * (
                score(np.delete(mixed, i, axis=0)) / score(np.delete(obs, i, axis=0))
                - 1
            )
            for i in range(8)
        ]
        for i, origin in enumerate(origins[0]):
            rows.append(
                dict(
                    updates=update,
                    origin=origin,
                    mixed_normalized_rmse=float(np.sqrt(mixed[i]).mean()),
                    obs_only_normalized_rmse=float(np.sqrt(obs[i]).mean()),
                    mixed_relative_error_percent=float(
                        100 * (np.sqrt(mixed[i]).mean() / np.sqrt(obs[i]).mean() - 1)
                    ),
                )
            )
        summaries[str(update)] = dict(
            relative_error_percent=float(100 * (score(mixed) / score(obs) - 1)),
            paired_origin_bootstrap_percentile95=np.percentile(
                bootstrap, [2.5, 97.5]
            ).tolist(),
            leave_one_origin_out_range=[min(leave_one_out), max(leave_one_out)],
            mixed_worse_origins=sum(
                r["mixed_relative_error_percent"] > 0
                for r in rows
                if r["updates"] == update
            ),
        )
    pd.DataFrame(rows).to_csv(output / "validation-origin-sensitivity.csv", index=False)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=Path(__file__).parent / "results"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = histories(args.results)
    agg = data.groupby(["arm", "updates"], as_index=False).first()
    merged = data[data.arm == "mixed"].merge(
        data[data.arm == "obs-only"],
        on=["updates", "lead_hours", "variable"],
        suffixes=("_mixed", "_obs"),
    )
    np.testing.assert_array_equal(
        merged.persistence_rmse_mixed, merged.persistence_rmse_obs
    )
    merged["mixed_relative_error_percent"] = 100 * (
        merged.model_rmse_mixed / merged.model_rmse_obs - 1
    )
    aggregate = agg.pivot(
        index="updates", columns="arm", values="aggregate_normalized_rmse"
    )
    aggregate["mixed_relative_error_percent"] = 100 * (
        aggregate["mixed"] / aggregate["obs-only"] - 1
    )
    aggregate.to_csv(args.output / "validation-training-aggregate.csv")
    merged[
        [
            "updates",
            "lead_hours",
            "variable",
            "model_rmse_mixed",
            "model_rmse_obs",
            "persistence_rmse_mixed",
            "mixed_relative_error_percent",
        ]
    ].to_csv(args.output / "validation-training-paired.csv", index=False)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.7), layout="constrained")
    persistence = agg.persistence_aggregate_normalized_rmse.iloc[0]
    for arm in COLORS:
        curve = agg[agg.arm == arm]
        for ax in axes:
            ax.plot(
                curve.updates,
                curve.aggregate_normalized_rmse,
                color=COLORS[arm],
                label=LABELS[arm],
                lw=1.7,
                marker=".",
                markersize=3,
            )
    for ax in axes:
        ax.axhline(
            persistence, color="#444444", linestyle="--", lw=1.3, label="Persistence"
        )
        shade(ax)
        ax.set_ylabel("OBS aggregate normalized RMSE (lower is better)")
    axes[0].set_yscale("log")
    axes[0].set_title("Full training history · logarithmic error axis")
    axes[0].set_xlim(0, 11100)
    axes[0].legend(loc="upper right", frameon=False)
    axes[1].set_xlim(8000, 11050)
    axes[1].set_ylim(0.018, 0.029)
    axes[1].set_title("Late training · linear error axis")
    axes[1].text(
        10040,
        0.0284,
        "Both arms: OBS-only cooldown\nEMA reset after 10k",
        fontsize=9,
        va="top",
    )
    axes[0].annotate(
        "Mixed spikes",
        xy=(6800, aggregate.loc[6800, "mixed"]),
        xytext=(7600, 0.12),
        arrowprops=dict(arrowstyle="-", color="#666666"),
        fontsize=10,
    )
    axes[0].annotate(
        "OBS-only spike",
        xy=(5000, aggregate.loc[5000, "obs-only"]),
        xytext=(2500, 0.25),
        arrowprops=dict(arrowstyle="-", color="#666666"),
        fontsize=10,
    )
    fig.suptitle(
        "Same eight validation origins · EMA every 100 updates · no additional smoothing",
        fontsize=13,
    )
    fig.savefig(args.output / "validation-training-overview.png", dpi=180)
    fig.savefig(args.output / "validation-training-overview.svg")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.3), layout="constrained")
    for row, variable in enumerate(("thetao_0", "zos")):
        unit, label = ("°C", "SST") if variable == "thetao_0" else ("cm", "SSH")
        scale = 1 if row == 0 else 100
        for col, lead in enumerate((24, 48, 72)):
            ax = axes[row, col]
            subset = data[(data.variable == variable) & (data.lead_hours == lead)]
            for arm in COLORS:
                curve = subset[subset.arm == arm]
                ax.plot(
                    curve.updates,
                    scale * curve.model_rmse,
                    color=COLORS[arm],
                    label=LABELS[arm],
                    lw=1.5,
                )
            ax.axhline(
                scale * subset.persistence_rmse.iloc[0],
                color="#444444",
                ls="--",
                lw=1.2,
                label="Persistence",
            )
            shade(ax)
            ax.set_xlim(2000, 11050)
            ax.set_yscale("log")
            ax.set_title(f"{label} · {lead} h")
            ax.set_ylabel(f"RMSE ({unit}; log scale)")
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Validation by field and lead · continuation from update 2,000 · shaded region: cooldown",
        fontsize=13,
    )
    fig.savefig(args.output / "validation-training-fields.png", dpi=180)
    fig.savefig(args.output / "validation-training-fields.svg")
    plt.close(fig)
    summary: dict = {}
    for name, start, end in [("late_main", 8100, 10000), ("cooldown", 10100, 11000)]:
        selected = merged[merged.updates.between(start, end)]
        gaps = aggregate.loc[start:end, "mixed_relative_error_percent"]
        summary[name] = dict(
            checkpoints=len(gaps),
            field_lead_comparisons=len(selected),
            mixed_worse=int((selected.mixed_relative_error_percent > 0).sum()),
            aggregate_gap_min=float(gaps.min()),
            aggregate_gap_median=float(gaps.median()),
            aggregate_gap_max=float(gaps.max()),
        )
    summary["origin_sensitivity"] = paired_summary(
        args.results / "validation-origin-mse.csv", agg, args.output
    )
    summary["source_sha256"] = {
        name: hashlib.sha256((args.results / name).read_bytes()).hexdigest()
        for name in (
            "mixed-observation-validation.csv",
            "mixed-long-observation-validation.csv",
            "obs-only-validation.csv",
            "obs-only-long-validation.csv",
            "validation-origin-mse.csv",
        )
    }
    (args.output / "validation-training-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
