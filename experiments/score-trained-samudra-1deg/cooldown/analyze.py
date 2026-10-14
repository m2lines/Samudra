# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Reproduce cooldown comparisons from unmodified job artifacts in --artifacts."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_score import (  # type: ignore[import-not-found]  # noqa: E402
    save_csv,
    save_figure,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("/tmp"))
    args = parser.parse_args()
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    metrics = [
        "rmse_normalized",
        "crps_empirical_normalized",
        "spread_normalized",
        "coverage_member_range",
    ]
    forecasts, completions, continuous, phases = [], [], [], []
    for arm, job, oldf, oldc, oldd in [
        ("F", 19815733, 19758710, 19762850, 19762854),
        ("M", 19815734, 19758711, 19762851, 19762855),
    ]:
        base = args.artifacts / f"eval-cooldown-{arm}-{job}"
        old = pd.read_csv(args.artifacts / f"eval-full-{arm}-{oldf}/metrics-rank0.csv")
        for role in ["before", "ema", "raw"]:
            f = (
                old.copy()
                if role == "before"
                else pd.read_csv(base / role / "metrics-rank0.csv")
            )
            assert len(f) == 16170 and np.isfinite(f[metrics]).all().all()
            keys = ["origin", "lead_days", "variable"]
            pd.testing.assert_frame_equal(old[keys], f[keys])
            summary = (
                f.groupby("lead_days")[metrics]
                .mean()
                .reset_index()
                .assign(arm=arm, role=role)
            )
            forecasts.append(summary)
            save_csv(
                f.groupby(["origin", "lead_days"])[metrics].mean().reset_index(),
                out / f"{arm}-{role}-forecast-origins.csv",
            )
        for role, cp, dp in [
            (
                "before",
                args.artifacts / f"eval-completion-{arm}-{oldc}",
                args.artifacts / f"eval-continuous-{arm}-{oldd}" / arm,
            ),
            ("ema", base / "completion", base / "continuous" / arm),
        ]:
            c = pd.read_csv(cp / "metrics.csv")
            assert len(c) == 9992 and np.isfinite(c[metrics]).all().all()
            origin = c.groupby(["task", "origin"])[metrics].mean().reset_index()
            completions.append(
                origin.groupby("task")[metrics]
                .mean()
                .reset_index()
                .assign(arm=arm, role=role)
            )
            s = pd.read_csv(dp / "scores.csv.gz")
            assert len(s) == 46046
            cols = [
                "rmse",
                "crps_empirical",
                "crps_fair",
                "spread",
                "coverage_member_range",
            ]
            assert np.isfinite(s[cols]).all().all()
            s["year"] = s.target.str[:4].astype(int)
            continuous.append(
                s.groupby("year")[cols].mean().reset_index().assign(arm=arm, role=role)
            )
            z = np.load(dp / "structure.npz")
            idx = list(z["variables"]).index("so_9")
            for parity, phase in [(0, "across calls"), (1, "within call")]:
                a = z["temporal_increment_rms"][parity::2, :, idx].mean(0)
                phases.append(
                    dict(
                        arm=arm,
                        role=role,
                        phase=phase,
                        truth=a[0],
                        members=a[2:].mean(),
                        ratio=a[2:].mean() / a[0],
                    )
                )
    f = pd.concat(forecasts)
    c = pd.concat(completions)
    d = pd.concat(continuous)
    p = pd.DataFrame(phases)
    for name, frame in [
        ("forecasts", f),
        ("completion", c),
        ("continuous", d),
        ("increments", p),
    ]:
        save_csv(frame, out / f"{name}.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for arm, color in [("F", "#0072b2"), ("M", "#d55e00")]:
        for role, marker in [("before", "o"), ("ema", "s"), ("raw", "^")]:
            a = f[(f.arm == arm) & (f.role == role)]
            for ax, metric in zip(axes, metrics[:2], strict=True):
                ax.plot(
                    a.lead_days,
                    a[metric],
                    color=color,
                    marker=marker,
                    label=f"{arm} {role}",
                )
                ax.set(
                    xlabel="Nominal lead (days)",
                    ylabel="Normalized RMSE"
                    if metric == metrics[0]
                    else "Normalized CRPS",
                )
                ax.grid(alpha=0.2)
                ax.legend(fontsize=8)
    save_figure(fig, out / "forecast.jpg")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for arm, color in [("F", "#0072b2"), ("M", "#d55e00")]:
        baseline = f[(f.arm == arm) & (f.role == "before")].set_index("lead_days")
        for role, marker in [("ema", "s"), ("raw", "^")]:
            values = f[(f.arm == arm) & (f.role == role)].set_index("lead_days")
            for ax, metric in zip(axes, metrics[:2], strict=True):
                delta = 100 * (values[metric] / baseline[metric] - 1)
                ax.plot(
                    delta.index,
                    delta,
                    color=color,
                    marker=marker,
                    label=f"{arm} {role}",
                )
                ax.set(
                    xlabel="Nominal lead (days)",
                    ylabel="Change vs own pre-cooldown EMA (%)",
                    title="RMSE" if metric == metrics[0] else "Empirical CRPS",
                )
                ax.legend()
                ax.grid(alpha=0.2)
    for ax in axes:
        ax.axhline(0, color="black", linewidth=0.7)
    save_figure(fig, out / "forecast-change.jpg")
    print(f[f.lead_days == 360].to_string(index=False))
    print(c.to_string(index=False))
    print(d[d.year == 2022].to_string(index=False))
    print(p.to_string(index=False))


if __name__ == "__main__":
    main()
