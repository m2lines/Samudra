# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Summarize the completed F/M wave from downloaded, unmodified evaluation folders."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analyze_score import save_csv, save_figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("/tmp"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    out = root / "results/multitask"
    out.mkdir(exist_ok=True)
    figures = root / "figures/multitask"
    figures.mkdir(parents=True, exist_ok=True)
    metrics = [
        "rmse_normalized",
        "crps_empirical_normalized",
        "spread_normalized",
        "coverage_q05_q95",
        "coverage_member_range",
    ]
    completions, continuous, phases, bulk = [], [], [], []
    forecast = []
    for arm, cj, dj, fj in [
        ("F", 19762850, 19762854, 19758710),
        ("M", 19762851, 19762855, 19758711),
    ]:
        cp = args.artifacts / f"eval-completion-{arm}-{cj}"
        dp = args.artifacts / f"eval-continuous-{arm}-{dj}" / arm
        f = pd.read_csv(args.artifacts / f"eval-full-{arm}-{fj}/metrics-rank0.csv")
        assert len(f) == 16170
        forecast.append(f)
        c = pd.read_csv(cp / "metrics.csv")
        assert len(c) == 9992 and np.isfinite(c[metrics]).all().all()
        origin = c.groupby(["arm", "task", "origin"])[metrics].mean().reset_index()
        save_csv(origin, out / f"{arm}-completion-by-origin.csv")
        completions.append(
            origin.groupby(["arm", "task"])[metrics].mean().reset_index()
        )
        s = pd.read_csv(dp / "scores.csv.gz")
        assert len(s) == 46046
        scorecols = [
            "rmse",
            "crps_empirical",
            "crps_fair",
            "spread",
            "coverage_q05_q95",
            "coverage_member_range",
        ]
        assert np.isfinite(s[scorecols]).all().all()
        dates = s.groupby(["arm", "target"])[scorecols].mean().reset_index()
        dates["year"] = dates.target.str[:4].astype(int)
        save_csv(dates, out / f"{arm}-continuous-by-date.csv")
        continuous.append(
            dates.groupby(["arm", "year"])[scorecols].mean().reset_index()
        )
        g = pd.read_csv(dp / "global.csv.gz")
        for slot, group in g.groupby("slot"):
            bulk.append(
                dict(
                    arm=arm,
                    slot=slot,
                    thetao_temporal_std=group.volume_mean_thetao.std(),
                    thetao_mean=group.volume_mean_thetao.mean(),
                    so_mean=group.volume_mean_so.mean(),
                )
            )
        z = np.load(dp / "structure.npz")
        idx = list(z["variables"]).index("so_9")
        for parity, name in [(0, "across calls"), (1, "within call")]:
            # The first increment compares the first output against the initial state.
            a = z["temporal_increment_rms"][parity::2, :, idx].mean(0)
            phases.append(
                dict(
                    arm=arm,
                    phase=name,
                    truth=a[0],
                    ensemble_mean=a[1],
                    members=a[2:].mean(),
                    member_truth_ratio=a[2:].mean() / a[0],
                )
            )
    completion = pd.concat(completions)
    save_csv(completion, out / "completion-summary.csv")
    yearly = pd.concat(continuous)
    save_csv(yearly, out / "continuous-yearly.csv")
    phase = pd.DataFrame(phases)
    save_csv(phase, out / "salinity-increments.csv")
    save_csv(pd.DataFrame(bulk), out / "bulk-summary.csv")
    f = pd.concat(forecast)
    assert not f.duplicated(["arm", "origin", "lead_days", "variable"]).any()
    summary = f.groupby(["arm", "lead_days"])[metrics].mean().reset_index()
    rng = np.random.default_rng(610001)
    intervals = []
    for lead, group in f.groupby("lead_days"):
        for metric in metrics[:2]:
            paired = group.pivot(
                index=["origin", "variable"], columns="arm", values=metric
            )
            assert paired.notna().all().all()
            paired["year"] = paired.index.get_level_values("origin").str[:4]
            blocks = paired.groupby("year")[["F", "M"]].sum().to_numpy()
            counts = paired.groupby("year").size().to_numpy()
            samples = rng.integers(0, len(blocks), (10000, len(blocks)))
            means = blocks[samples].sum(1) / counts[samples].sum(1)[:, None]
            differences = means[:, 1] - means[:, 0]
            lo, hi = np.quantile(differences, [0.025, 0.975])
            intervals.append(
                dict(
                    lead_days=lead,
                    metric=metric,
                    delta_M_minus_F=(paired.M - paired.F).mean(),
                    low=lo,
                    high=hi,
                )
            )
    save_csv(pd.DataFrame(intervals), out / "forecast-year-block-intervals.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    for arm, color, marker in [("F", "#0072b2", "o"), ("M", "#d55e00", "s")]:
        a = summary[summary.arm == arm]
        for ax, metric, title in zip(
            axes,
            metrics[:3],
            ["Ensemble-mean RMSE", "Empirical CRPS", "Ensemble spread"],
            strict=True,
        ):
            ax.plot(a.lead_days, a[metric], color=color, marker=marker, label=arm)
            ax.set(xlabel="Lead (nominal days)", ylabel="Normalized units", title=title)
            ax.legend()
            ax.grid(alpha=0.2)
    save_figure(fig, figures / "forecast.jpg")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    for arm, color, marker in [("F", "#0072b2", "o"), ("M", "#d55e00", "s")]:
        a = yearly[yearly.arm == arm]
        axes[0].plot(a.year, a.rmse, color=color, marker=marker, label=arm)
        a = phase[phase.arm == arm]
        axes[1].plot(
            a.phase, a.member_truth_ratio, color=color, marker=marker, label=arm
        )
    axes[0].set(
        title="Continuous rollout (4 members)",
        ylabel="Normalized RMSE",
        xlabel="Calendar year",
    )
    axes[1].set(
        title="Salinity so_9 temporal increments", ylabel="Member RMS increment / truth"
    )
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    save_figure(fig, figures / "continuous.jpg")
    # Exactly 360 x 180 rendered pixels per map: one pixel per model cell.
    a = np.load(args.artifacts / "eval-completion-F-19762850/fields-surface.npz")
    b = np.load(args.artifacts / "eval-completion-M-19762851/fields-surface.npz")
    np.testing.assert_array_equal(a["target"], b["target"])
    np.testing.assert_array_equal(a["request"], b["request"])
    i = 2  # first time, so_9; channels saved as zos, thetao_0, so_9, uo_9
    mask = b["request"][i].astype(bool)
    lo, hi = np.quantile(b["target"][i][mask], [0.01, 0.99])
    fig = plt.figure(figsize=(8, 8), dpi=100)
    for n, (label, field) in enumerate(
        [
            ("OM4 truth", b["target"][i]),
            ("F mean", a["mean"][i]),
            ("M mean", b["mean"][i]),
            ("M member 1", b["member0"][i]),
            ("M member 2", b["member1"][i]),
            ("M spread", b["spread"][i]),
        ]
    ):
        col, row = n % 2, n // 2
        ax = fig.add_axes(
            ((30 + 400 * col) / 800, (540 - 220 * row) / 800, 360 / 800, 180 / 800)
        )
        im = ax.imshow(
            np.where(mask, field, np.nan),
            origin="lower",
            interpolation="nearest",
            vmin=0 if n == 5 else lo,
            vmax=float(np.quantile(field[mask], 0.99)) if n == 5 else hi,
            cmap="magma" if n == 5 else "viridis",
        )
        ax.set_title(label, fontsize=10)
        ax.set_axis_off()
        if n in (4, 5):
            cb = fig.colorbar(
                im,
                cax=fig.add_axes(((30 + 400 * col) / 800, 0.055, 360 / 800, 0.015)),
                orientation="horizontal",
            )
            cb.set_label("Normalized salinity" if n == 4 else "Normalized spread")
    fig.suptitle(
        "Surface → interior: so_9, first held-out origin, first date\nNormalized salinity; common truth 1–99% limits except spread",
        fontsize=11,
    )
    save_figure(fig, figures / "surface-so9.jpg")
    print(completion.to_string(index=False))
    print(pd.DataFrame(intervals).tail(2).to_string(index=False))


if __name__ == "__main__":
    main()
