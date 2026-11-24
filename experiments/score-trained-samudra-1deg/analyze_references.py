# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit training-only references against the frozen primary annual comparison."""

import argparse
import gzip
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_score import LICENSE, STYLES, plt, save_csv, save_figure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("references", type=Path)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.references / "manifest.json").read_text())
    complete = json.loads((args.references / "COMPLETE.json").read_text())
    assert complete == {"rows": 32340, "origins": 30}
    assert manifest["train_first"] == "1975-01-03 12:00:00"
    assert manifest["train_last"] == "2013-10-05 12:00:00"
    assert manifest["train_count"] == 2830
    # 1975--2012 complete, plus the first 56 five-day centers of 2013.
    assert manifest["seasonal_bin_counts"] == [39] * 56 + [38] * 17
    assert sum(manifest["seasonal_bin_counts"]) == manifest["train_count"]
    data = pd.concat(
        [pd.read_csv(p) for p in sorted(args.references.glob("metrics-rank*.csv"))],
        ignore_index=True,
    )
    keys = ["arm", "origin", "lead_days", "variable"]
    actual = set(data[keys].itertuples(index=False, name=None))
    expected = set(
        itertools.product(
            "PS",
            manifest["origin_times"],
            manifest["leads_days"],
            manifest["variables"],
        )
    )
    assert len(actual) == len(data) == 32340 and actual == expected
    assert np.isfinite(data.select_dtypes("number")).all().all()
    assert data.members.eq(1).all() and data.spread_normalized.eq(0).all()
    np.testing.assert_array_equal(data.crps_empirical_normalized, data.mae_normalized)
    np.testing.assert_array_equal(data.crps_fair_normalized, data.mae_normalized)
    np.testing.assert_array_equal(data.member_rmse_normalized, data.rmse_normalized)
    scale = data.variable.map(
        dict(zip(manifest["variables"], manifest["normalization_std"], strict=True))
    )
    for column in map(str, data.columns):
        if column.endswith("_normalized"):
            np.testing.assert_allclose(
                data[column.replace("_normalized", "_physical")],
                data[column] * scale,
                rtol=2e-6,
                atol=1e-8,
            )
    primary = pd.concat(
        [
            pd.read_csv(p)
            for p in sorted((args.results / "annual-raw").glob("*.csv.gz"))
        ],
        ignore_index=True,
    )
    dates = primary[primary.arm == "A"].set_index(keys[1:]).target.sort_index()
    for arm in "PS":
        pd.testing.assert_series_equal(
            data[data.arm == arm].set_index(keys[1:]).target.sort_index(), dates
        )
    assert manifest["train_last"] < min(manifest["origin_times"])
    contract = json.loads((args.results / "annual-contract.json").read_text())
    for key in ("variables", "normalization_std", "origin_times", "leads_days"):
        assert manifest[key] == contract[key]
    raw_dir = args.results / "reference-raw"
    raw_dir.mkdir(exist_ok=True)
    for (arm, lead), frame in data.groupby(["arm", "lead_days"]):
        ordered = frame.sort_values(["origin", "variable"])
        for part, ids in enumerate(np.array_split(np.arange(len(ordered)), 3)):
            path = raw_dir / f"{arm}-{lead:03d}d-{part}.csv.gz"
            path.write_bytes(
                gzip.compress(ordered.iloc[ids].to_csv(index=False).encode(), mtime=0)
            )
            assert path.stat().st_size < 250 * 1024
            path.with_suffix(path.suffix + ".license").write_text(LICENSE)
    combined = pd.concat([primary, data], ignore_index=True)
    combined["family"] = combined.variable.str.split("_").str[0]
    metrics = [k for k in map(str, data.columns) if k.endswith("_normalized")]
    summary = combined.groupby(["arm", "lead_days"])[metrics].mean().reset_index()
    save_csv(summary, args.results / "reference-comparison.csv")
    family = (
        combined.groupby(["arm", "lead_days", "family"])[
            ["rmse_normalized", "rmse_physical", "crps_empirical_physical"]
        ]
        .mean()
        .reset_index()
    )
    save_csv(family, args.results / "reference-family.csv")
    channel = data.groupby(["arm", "lead_days", "variable"])[
        ["rmse_normalized", "rmse_physical", "crps_empirical_physical"]
    ].mean()
    save_csv(channel.reset_index(), args.results / "reference-channel.csv")
    styles = {**STYLES, "P": ("#8c65a6", "x"), "S": ("#b59b00", "P")}
    labels = {"P": "P: persistence", "S": "S: seasonal climatology"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, metric, title in zip(
        axes,
        ("rmse_normalized", "crps_empirical_normalized"),
        ("Ensemble-mean RMSE", "Empirical CRPS"),
        strict=True,
    ):
        for plot_arm, group in summary.groupby("arm"):
            color, marker = styles[str(plot_arm)]
            ax.plot(
                group.lead_days,
                group[metric],
                color=color,
                marker=marker,
                label=labels.get(str(plot_arm), str(plot_arm)),
            )
        ax.set(
            xlabel="Nominal lead (days)",
            ylabel="Mean normalized channel score",
            title=title,
        )
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    save_figure(fig, args.results / "reference-comparison.jpg")
    path = args.results / "reference-manifest.json.gz"
    path.write_bytes(gzip.compress(json.dumps(manifest, indent=2).encode(), mtime=0))
    path.with_suffix(path.suffix + ".license").write_text(LICENSE)
    print(
        "Reference audit passed: 32,340 rows, training-only bins, common annual targets."
    )
    print(summary[summary.lead_days == 360].to_string(index=False))
    print(family[family.lead_days == 360].to_string(index=False))


if __name__ == "__main__":
    main()
