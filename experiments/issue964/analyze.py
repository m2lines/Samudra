# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
import csv
import gzip
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = pathlib.Path("experiments/issue964/results")
arms = ["oct24", "legacy_mse", "wet_mse", "dynamic_wet"]
names = ["Oct24 dynamic legacy", "Legacy MSE", "Wet-cell MSE", "Dynamic wet-cell MSE"]
key = "rollout_val/360d/normalized_rmse/channel_mean"
rows = []
depth = []
fig, ax = plt.subplots(figsize=(7, 4))
for a, n in zip(arms, names):
    x = json.load(gzip.open(p / f"{a}-wandb.json.gz"))
    h = x["history"]
    best = min(h, key=lambda r: r[key])
    final = max(h, key=lambda r: r["epoch"])
    ev = {r["label"]: r["metrics"] for r in json.load(open(p / f"{a}-eval.json"))}
    r = {
        "arm": a,
        "best_epoch": best["epoch"],
        "best_360d": best[key],
        "final_360d": final[key],
        "final_ema_time_mean": ev["ema_latest"][
            "inference/time_mean_norm/rmse/channel_mean"
        ],
        "final_raw_time_mean": ev["epoch_0070"][
            "inference/time_mean_norm/rmse/channel_mean"
        ],
    }
    for v in ["thetao", "so", "uo", "vo", "zos"]:
        k = f"rollout_val/360d/weighted_rmse/{v}" + (
            "/mean_depths" if v != "zos" else ""
        )
        r[v] = final[k]
    for selection, d in [("best", best), ("final", final)]:
        for k, v in d.items():
            if "/weighted_rmse/" in k:
                depth.append(
                    {
                        "arm": a,
                        "selection": selection,
                        "epoch": d["epoch"],
                        "metric": k,
                        "value": v,
                    }
                )
    rows.append(r)
    ax.plot([t["epoch"] for t in h], [t[key] for t in h], label=n)
ax.set(
    xlabel="Epoch",
    ylabel="360-day normalized RMSE (lower is better)",
    ylim=(0.27, 0.40),
)
ax.legend(fontsize=8)
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(p / "validation.png", dpi=160)
for fn, data in [("comparison.csv", rows), ("physical-depth.csv", depth)]:
    with open(p / fn, "w") as f:
        w = csv.DictWriter(f, fieldnames=data[0].keys())
        w.writeheader()
        w.writerows(data)
for r in rows:
    print(r)
