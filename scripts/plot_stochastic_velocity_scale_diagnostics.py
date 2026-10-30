# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root = args.input
out = args.output
rows = []
for p in root.glob("*-02000-*.json"):
    r = json.loads(p.read_text())
    for b in r["speed_bins"]:
        rows.append(
            dict(
                variant=r["variant"],
                month=r["month"],
                low=b["low"],
                high=b["high"],
                area=sum(b["area"]),
                squared_error_sum=sum(b["squared_error_sum"]),
                predicted_power_sum=sum(b["predicted_power_sum"]),
                gold_power_sum=sum(b["gold_power_sum"]),
            )
        )
d = pd.DataFrame(rows)
d.to_csv(out / "current-strength-by-date.csv", index=False)
a = d.groupby(["variant", "low"], as_index=False)[
    ["area", "squared_error_sum", "predicted_power_sum", "gold_power_sum"]
].sum()
a["rmse"] = np.sqrt(a.squared_error_sum / a.area)
a["rmse_vs_zero"] = np.sqrt(a.squared_error_sum / a.gold_power_sum)
a["predicted_gold_power_ratio"] = a.predicted_power_sum / a.gold_power_sum
a.to_csv(out / "current-strength-aggregate.csv", index=False)
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
for variant, rr in a.groupby("variant"):
    for ax, k in zip(axes, ["rmse_vs_zero", "predicted_gold_power_ratio"]):
        ax.plot(range(6), rr[k], marker="o", label=variant)
        ax.set_xticks(
            range(6), ["0–.01", ".01–.03", ".03–.1", ".1–.3", ".3–1", ">1"], rotation=30
        )
        ax.set_xlabel("Reference vector speed (m/s)")
        ax.set_title(
            "RMSE relative to zero-velocity prediction"
            if k == "rmse_vs_zero"
            else "Predicted / reference squared velocity"
        )
        ax.grid(alpha=0.2)
axes[0].legend(fontsize=7)
axes[1].set_yscale("log")
for ax in axes:
    ax.axhline(1, color="black", linewidth=0.8, alpha=0.6)
fig.savefig(out / "current-strength.png", dpi=150)
fig.savefig(out / "current-strength.pdf")
plt.close(fig)
s = pd.read_csv(out / "pacific-spectra.csv")
print("Pacific minimumwetfraction", s.wet_fraction.min())
fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
for ax, c in zip(axes, ["uo_0", "vo_0"]):
    for variant, rr in s[s.channel == c].groupby("variant"):
        avg = rr.groupby("k_low")[["gold", "mean8", "mean_member_power"]].mean()
        frequencies = avg.index.to_numpy()
        ax.loglog(
            frequencies[1:],
            avg.mean_member_power.to_numpy()[1:],
            marker="o",
            markersize=3,
            label=variant,
        )
    ax.loglog(
        frequencies[1:],
        avg.gold.to_numpy()[1:],
        color="black",
        linewidth=2,
        label="OM4 reference",
    )
    ax.set_title(c + " average member spectrum")
    ax.set_xlabel("cycles/grid cell")
    ax.set_ylabel("tapered squared Fourier amplitude")
    ax.grid(alpha=0.2)
axes[0].legend(fontsize=7)
fig.savefig(out / "pacific-member-spectra.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
for variant in sorted(a.variant.unique()):
    cases = [json.loads(p.read_text()) for p in root.glob(f"{variant}-02000-*.json")]
    mse = np.mean([c["metrics"]["mean_mse"] for c in cases], axis=0)
    for ax, ids, title in zip(axes, [range(19), range(19, 38)], ["u", "v"]):
        ax.plot(
            np.sqrt(mse[list(ids)]), range(19), marker="o", markersize=3, label=variant
        )
        ax.set(
            xlabel="Ensemble-mean RMSE (m/s)", ylabel="Depth level index", title=title
        )
        ax.set_ylim(18, 0)
        ax.grid(alpha=0.2)
axes[0].legend(fontsize=7)
fig.savefig(out / "depth-errors.png", dpi=150)
plt.close(fig)
