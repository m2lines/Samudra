from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path("experiments/hfds-anomaly/results")
rng = np.random.default_rng(15)
summary = []
channel_summary = []
for selection in ["selected", "final-ema"]:
    d = pd.concat(
        [
            pd.read_csv(p)
            for p in sorted(
                (root / "scores" / selection / "lead_metrics").glob("*.csv")
            )
        ],
        ignore_index=True,
    )
    assert len(d) == 2 * 28 * 77 * 7
    assert np.isfinite(d[["rmse", "bias", "normalized_rmse"]]).all().all()
    assert not d.duplicated(["arm", "origin", "channel", "nominal_days"]).any()
    d["variable"] = d.channel.str.split("_").str[0]
    for variable in ["all", "thetao", "so", "uo", "vo", "zos"]:
        sub = d if variable == "all" else d[d.variable == variable]
        g = (
            sub.groupby(["origin", "nominal_days", "arm"])
            .normalized_rmse.mean()
            .unstack("arm")
            .reset_index()
        )
        g["year"] = g.origin.str[:4]
        for lead, x in g.groupby("nominal_days"):
            annual = x.groupby("year")[["with", "without"]].mean().to_numpy()
            samples = annual[rng.integers(0, len(annual), (10000, len(annual)))].mean(
                axis=1
            )
            boot = 100 * (samples[:, 1] / samples[:, 0] - 1)
            means = annual.mean(axis=0)
            summary.append(
                dict(
                    selection=selection,
                    variable=variable,
                    nominal_days=lead,
                    with_rmse=means[0],
                    without_rmse=means[1],
                    delta_percent=100 * (means[1] / means[0] - 1),
                    ci_low=np.quantile(boot, 0.025),
                    ci_high=np.quantile(boot, 0.975),
                )
            )
    c = (
        d.groupby(["channel", "nominal_days", "arm"])[
            ["rmse", "bias", "normalized_rmse"]
        ]
        .mean()
        .unstack("arm")
    )
    c.columns = ["_".join(k) for k in c.columns]
    c["rmse_delta_percent"] = 100 * (c.rmse_without / c.rmse_with - 1)
    c.reset_index().to_csv(
        root / "scores" / selection / "channel_summary.csv", index=False
    )
    print(
        selection,
        "actual lead ranges",
        d.groupby("nominal_days").actual_days.agg(["min", "max"]).to_dict("index"),
    )
s = pd.DataFrame(summary)
s.to_csv(root / "lead_summary.csv", index=False)
print(s[s.variable == "all"].to_string(index=False))
fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, (label, g) in zip(axs, s[s.variable == "all"].groupby("selection", sort=False)):
    ax.plot(g.nominal_days, g.delta_percent, "o-")
    ax.fill_between(g.nominal_days, g.ci_low, g.ci_high, alpha=0.2)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(g.nominal_days, g.nominal_days)
    ax.set_title(label)
    ax.set_xlabel("Nominal lead (days)")
    ax.grid(alpha=0.2)
axs[0].set_ylabel("No-anomaly change in mean normalized RMSE (%)")
fig.suptitle("Negative favors removing hfds_anoms; 95% paired origin-year bootstrap")
fig.tight_layout()
fig.savefig(root / "lead_comparison.png", dpi=170)
obs = []
for label in ["selected", "final-ema"]:
    d = pd.read_csv(
        root / "viz" / label / "Figures_observations/observation_metrics.csv"
    ).fillna({"depth": ""})
    for (metric, depth), g in d.groupby(["metric", "depth"]):
        primary = g[g.period_kind == "primary_complete_years"].set_index("model").value
        if not {"with", "without"}.issubset(primary.index):
            continue
        annual = (
            g[g.period_kind == "annual"]
            .pivot(index="year", columns="model", values="value")[["with", "without"]]
            .dropna()
            .to_numpy()
        )
        draws = annual[rng.integers(0, len(annual), (10000, len(annual)))].mean(axis=1)
        boot = 100 * (draws[:, 1] / draws[:, 0] - 1)
        obs.append(
            dict(
                selection=label,
                metric=metric,
                depth=depth,
                with_rmse=primary["with"],
                without_rmse=primary["without"],
                delta_percent=100 * (primary["without"] / primary["with"] - 1),
                annual_mean_delta_percent=100
                * (annual[:, 1].mean() / annual[:, 0].mean() - 1),
                annual_ci_low=np.quantile(boot, 0.025),
                annual_ci_high=np.quantile(boot, 0.975),
                years=len(annual),
            )
        )
o = pd.DataFrame(obs)
o.to_csv(root / "observation_summary.csv", index=False)
print(o.to_string(index=False))
