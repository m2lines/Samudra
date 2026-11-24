# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit observation scores; keep metrics of means separate from member metrics."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_score import LICENSE, STYLES, plt, save_csv, save_figure

METRICS = [
    ("surface_geostrophic_velocity_vector_total_rmse", "", "Velocity", "m/s", 1),
    ("instantaneous_surface_eke_total_rmse", "", "EKE", "m²/s²", 1),
    ("surface_sst_total_rmse", "", "SST", "°C", 1),
    ("ohc_per_area_total_rmse", "Surface (0-700m)", "OHC 0–700 m", "GJ/m²", 1e9),
    (
        "ohc_per_area_total_rmse",
        "Intermediate (700-2000m)",
        "OHC 700–2000 m",
        "GJ/m²",
        1e9,
    ),
]
VARIANCE = [
    ("surface_sst_residual_variance_map_rmse", "SST variance-map RMSE", "°C²", 1),
    ("surface_sst_residual_variance_pattern_corr", "SST pattern correlation", "1", 1),
    (
        "ohc_upper700_per_area_residual_variance_map_rmse",
        "Upper OHC variance-map RMSE",
        "(GJ/m²)²",
        1e18,
    ),
    (
        "ohc_upper700_per_area_residual_variance_pattern_corr",
        "Upper OHC pattern correlation",
        "1",
        1,
    ),
]


def subset(frame, metric, depth=""):
    return frame[(frame.metric == metric) & (frame.depth == depth)]


def audit(frame, manifest, complete, contract):
    labels = ["A-mean", "B-mean"] + [
        f"{arm}-{slot}"
        for arm in "CD"
        for slot in ["mean"] + [f"member{i}" for i in range(4)]
    ]
    assert complete == {"rows": 612, "models": labels}
    assert len(frame) == 612 and set(frame.model) == set(labels)
    assert not frame.duplicated(
        ["model", "metric", "depth", "period_kind", "year"]
    ).any()
    assert np.isfinite(frame.value).all()
    assert manifest["slurm_job"] == "132094"
    assert manifest["source_commit"].startswith("71c226400")
    for arm, record in manifest["rollout_manifests"].items():
        assert record["checkpoint_epoch"] == 70 and record["slurm_job"] == "212433"
        assert record["checkpoint"] == contract["arms"][arm]["checkpoint"]
        assert record["variables"] == contract["variables"]
        assert record["normalization_std"] == contract["normalization_std"]
        assert record["members"] == (1 if arm in "AB" else 4)
        assert record["origin"] == "2014-10-15 12:00:00"
        assert record["last_target"] == "2022-12-24 12:00:00"
        assert not record["partial"] and record["replay_verified_against"]
    for product, total, scored in (
        ("duacs", 598, 583),
        ("oisst", 598, 583),
        ("argo", 97, 84),
    ):
        assert manifest["products"][product]["full_common_samples"] == total
        assert manifest["products"][product]["scored_samples"] == scored
    primary = frame[frame.period_kind == "primary_complete_years"]
    annual = frame[frame.period_kind == "annual"]
    diagnostics = frame[frame.period_kind == "full_overlap"]
    assert primary.groupby("model").size().eq(5).all()
    assert annual.groupby("model").size().eq(38).all()
    assert diagnostics.groupby("model").size().eq(8).all()
    for metric, depth, _, _, _ in METRICS:
        p, a = subset(primary, metric, depth), subset(annual, metric, depth)
        is_ohc = bool(depth)
        years = list(range(2015, 2022 if is_ohc else 2023))
        dates = ("2015-01-01", "2021-12-01") if is_ohc else ("2015-01-03", "2022-12-24")
        assert p.period_start.eq(dates[0]).all() and p.period_end.eq(dates[1]).all()
        assert p.n_time_samples.eq(84 if is_ohc else 583).all()
        assert p.n_years.eq(len(years)).all() and p.n_bootstrap.eq(10000).all()
        assert p.uncertainty_method.eq("calendar_year_block_bootstrap").all()
        for column in (
            "grid_shape",
            "units",
            "n_paired_cells",
            "paired_ocean_fraction",
        ):
            assert p[column].nunique() == 1 and p[column].notna().all()
        assert p.paired_ocean_fraction.between(0, 1).all()
        for label in labels:
            rows = a[a.model == label].sort_values("year")
            assert rows.year.tolist() == years
            assert rows.n_time_samples.tolist() == (
                [12] * 7 if is_ohc else [73] * 7 + [72]
            )
            row = p[p.model == label].iloc[0]
            np.testing.assert_allclose(
                np.sqrt(np.mean(rows.value**2)), row.value, rtol=1e-12
            )
            assert np.isfinite(
                row[["annual_std", "ci_low", "ci_high", "map_weighted_rmse"]].to_numpy(
                    dtype=float
                )
            ).all()
            assert row.ci_low <= row.value <= row.ci_high
            np.testing.assert_allclose(
                rows.value.std(ddof=1), row.annual_std, rtol=1e-12
            )
    for metric, _, _, _ in VARIANCE:
        rows = diagnostics[diagnostics.metric == metric]
        is_ohc = metric.startswith("ohc")
        assert set(rows.model) == set(labels)
        assert rows.n_time_samples.eq(97 if is_ohc else 598).all()
        assert rows.period_start.eq("2014-11-01" if is_ohc else "2014-10-20").all()
        assert rows.period_end.eq("2022-11-01" if is_ohc else "2022-12-24").all()
        if metric.endswith("corr"):
            assert rows.value.between(-1, 1).all()
    for metric in ("ohc_bathymetry_disagreement", "ohc_partial_column_fraction"):
        for depth in (METRICS[3][1], METRICS[4][1]):
            assert subset(diagnostics, metric, depth).value.nunique() == 1
    return primary, annual, diagnostics


def paired_intervals(annual):
    """Resample shared calendar years and reconstruct root mean annual MSE."""
    rng = np.random.default_rng(610003)
    output = []
    for metric, depth, _, _, _ in METRICS:
        p = subset(annual, metric, depth).pivot(
            index="year", columns="model", values="value"
        )
        draws = rng.integers(len(p), size=(10000, len(p)))
        for arm, reference in (
            ("B", "A"),
            ("C", "A"),
            ("C", "B"),
            ("D", "A"),
            ("D", "B"),
            ("D", "C"),
        ):
            x, y = (
                p[f"{arm}-mean"].to_numpy() ** 2,
                p[f"{reference}-mean"].to_numpy() ** 2,
            )
            samples = 100 * (np.sqrt(x[draws].mean(1) / y[draws].mean(1)) - 1)
            low, high = np.quantile(samples, [0.025, 0.975])
            output.append(
                {
                    "metric": metric,
                    "depth": depth,
                    "arm": arm,
                    "reference": reference,
                    "relative_percent": 100 * (np.sqrt(x.mean() / y.mean()) - 1),
                    "ci_low_percent": low,
                    "ci_high_percent": high,
                    "n_years": len(p),
                    "n_bootstrap": 10000,
                    "seed": 610003,
                }
            )
    return pd.DataFrame(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.input / "observation_metrics.csv").fillna({"depth": ""})
    manifest = json.loads((args.input / "manifest.json").read_text())
    complete = json.loads((args.input / "COMPLETE.json").read_text())
    contract = json.loads((args.results / "annual-contract.json").read_text())
    primary, annual, diagnostics = audit(frame, manifest, complete, contract)
    intervals = paired_intervals(annual)
    save_csv(primary, args.results / "observation-primary.csv")
    save_csv(intervals, args.results / "observation-paired-year-intervals.csv")
    records = []
    for (metric, depth), group in frame[frame.period_kind != "annual"].groupby(
        ["metric", "depth"]
    ):
        for arm in "CD":
            members = group[group.model.str.startswith(f"{arm}-member")].value
            mean = group[group.model == f"{arm}-mean"].value.item()
            records.append(
                {
                    "metric": metric,
                    "depth": depth,
                    "arm": arm,
                    "metric_of_mean_field": mean,
                    "mean_member_metric": members.mean(),
                    "min_member_metric": members.min(),
                    "max_member_metric": members.max(),
                    "units": group.units.iloc[0],
                    "members": 4,
                }
            )
    save_csv(pd.DataFrame(records), args.results / "observation-member-summary.csv")

    # Exact raw bytes, with reversible compression and separate provenance.
    hashes = {}
    for source, target in (
        ("observation_metrics.csv", "observation-raw.csv.gz"),
        ("manifest.json", "observation-manifest.json.gz"),
        ("COMPLETE.json", "observation-complete.json.gz"),
    ):
        content = (args.input / source).read_bytes()
        hashes[source] = hashlib.sha256(content).hexdigest()
        path = args.results / target
        path.write_bytes(gzip.compress(content, mtime=0))
        assert gzip.decompress(path.read_bytes()) == content
        path.with_suffix(path.suffix + ".license").write_text(LICENSE)
    path = args.results / "observation-audit.json.gz"
    path.write_bytes(
        gzip.compress(
            json.dumps(
                {
                    "rows": len(frame),
                    "models": complete["models"],
                    "input_sha256": hashes,
                    "source_job": manifest["slurm_job"],
                    "aggregation": "root mean annual MSE, equal calendar years",
                    "paired_bootstrap": "10000 shared calendar-year resamples; not seed uncertainty",
                    "unscored_baseline": "obs preset baselines=[om4] is unused by direct score_rollouts; only 12 listed trajectories scored",
                },
                indent=2,
            ).encode(),
            mtime=0,
        )
    )
    path.with_suffix(path.suffix + ".license").write_text(LICENSE)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for ax, (metric, depth, title, units, scale) in zip(
        axes.flat, METRICS, strict=False
    ):
        rows = subset(annual, metric, depth)
        for arm in "ABCD":
            r = rows[rows.model == f"{arm}-mean"].sort_values("year")
            color, marker = STYLES[arm]
            ax.plot(r.year, r.value / scale, color=color, marker=marker, label=arm)
        ax.set(title=title, xlabel="Verification year", ylabel=f"RMSE ({units})")
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.grid(alpha=0.2)
    axes.flat[0].legend(ncol=2)
    axes.flat[5].axis("off")
    axes.flat[5].text(
        0,
        0.9,
        "A/B: deterministic\nC/D: metric of four-member mean\n\nOne continuous forecast from Oct 2014.\nSurface: 2015–2022, 583 five-day samples.\nOHC: 2015–2021, 84 monthly means.\n\nLower RMSE is better.\nFinal epoch-70 EMA; one training seed.",
        va="top",
        fontsize=12,
    )
    fig.suptitle("Observation RMSE by verification year", fontsize=16)
    save_figure(fig, args.results / "observation-annual.jpg")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for ax, (metric, depth, title, _, _) in zip(axes.flat, METRICS, strict=False):
        p = subset(primary, metric, depth).set_index("model").value
        baseline = p["A-mean"]
        for i, arm in enumerate("BCD"):
            row = subset(intervals, metric, depth)
            row = row[(row.arm == arm) & (row.reference == "A")].iloc[0]
            color, marker = STYLES[arm]
            ax.hlines(
                i, row.ci_low_percent, row.ci_high_percent, color=color, linewidth=2
            )
            ax.scatter(
                row.relative_percent, i, color=color, marker=marker, s=65, zorder=3
            )
        for i, arm in enumerate("CD", start=3):
            values = 100 * (
                p[[f"{arm}-member{j}" for j in range(4)]].to_numpy() / baseline - 1
            )
            color, _ = STYLES[arm]
            ax.scatter(values, np.full(4, i), color=color, marker="x", s=55)
            ax.scatter(
                values.mean(), i, facecolors="none", edgecolors=color, marker="o", s=100
            )
        ax.axvline(0, color="#666666", linewidth=1)
        ax.set(
            title=title,
            yticks=range(5),
            yticklabels=["B", "C mean", "D mean", "C members", "D members"],
            xlabel="RMSE change from A (%)",
        )
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.2)
    axes.flat[5].axis("off")
    axes.flat[5].text(
        0,
        0.9,
        "Mean-field rows: solid marker + 95%\npaired calendar-year bootstrap interval.\n\nMember rows: crosses are each member;\nhollow circle is average member RMSE.\nNo member confidence interval is implied.\n\nNegative is better than A.\nOnly 7–8 years of one forecast;\nintervals are not training-seed uncertainty.",
        va="top",
        fontsize=12,
    )
    fig.suptitle(
        "Observation RMSE: ensemble means and members answer different questions",
        fontsize=15,
    )
    save_figure(fig, args.results / "observation-comparison.jpg")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (metric, title, units, scale) in zip(axes.flat, VARIANCE, strict=True):
        rows = (
            diagnostics[diagnostics.metric == metric].set_index("model").value / scale
        )
        for i, arm in enumerate("ABCD"):
            color, marker = STYLES[arm]
            ax.scatter(i, rows[f"{arm}-mean"], marker=marker, color=color, s=70)
        for i, arm in enumerate("CD", start=4):
            v = rows[[f"{arm}-member{j}" for j in range(4)]].to_numpy()
            color, _ = STYLES[arm]
            ax.scatter(
                i + np.linspace(-0.09, 0.09, 4), v, color=color, marker="x", s=50
            )
            ax.scatter(
                i, v.mean(), facecolors="none", edgecolors=color, marker="o", s=100
            )
        direction = "higher is better" if metric.endswith("corr") else "lower is better"
        ax.set(
            title=f"{title}\n{direction}",
            ylabel=units,
            xticks=range(6),
            xticklabels=["A", "B", "C mean", "D mean", "C members", "D members"],
        )
        ax.tick_params(axis="x", labelrotation=15)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Detrended, deseasonalized variance: full common span\nFilled: mean-field metric; crosses: members; hollow: average member metric",
        fontsize=14,
    )
    save_figure(fig, args.results / "observation-variance.jpg")
    print(
        f"Audited {len(frame)} rows, 12 trajectories, 5 primary metrics and 8 diagnostics each."
    )
    print(intervals[intervals.reference == "A"].to_string(index=False))


if __name__ == "__main__":
    main()
