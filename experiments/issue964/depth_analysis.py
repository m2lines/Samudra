# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Export final logged depth losses and fixed rollout errors for issue 964."""

import csv
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

from samudra.constants import build_om4_layout

ROOT = Path(__file__).parent / "results"
ARMS = {
    "oct24": "6j1z2mwc",
    "legacy_mse": "39pgb6sm",
    "wet_mse": "h1and7h1",
    "dynamic_wet": "bbaw6yyn",
}
LABELS = {
    "oct24": "Dynamic legacy (Oct24)",
    "legacy_mse": "Legacy MSE",
    "wet_mse": "Wet-cell MSE",
    "dynamic_wet": "Dynamic wet-cell MSE",
}
COLORS = dict(zip(ARMS, ["#333333", "#3366aa", "#e68a22", "#009988"]))
DEPTHS = build_om4_layout(
    prognostic_vars_key="thermo_dynamic_all", boundary_vars_key="tau_hfds"
).depth_levels
VARS = {
    "thetao": ("Temperature", "°C"),
    "so": ("Salinity", "psu"),
    "uo": ("Eastward velocity", "m/s"),
    "vo": ("Northward velocity", "m/s"),
}
api = wandb.Api(timeout=60)
rows = []
raw = {}
rollouts = {}
for arm, rid in ARMS.items():
    run = api.run(f"ocean_emulators/default/{rid}")
    s = dict(run.summary)
    assert s["epoch"] == 70 and run.state == "finished", (
        arm,
        run.state,
        s.get("epoch"),
    )
    raw[arm] = {
        k: v
        for k, v in s.items()
        if "/loss/" in k or "/loss_scale/" in k or k == "epoch"
    }
    archive = json.load(gzip.open(ROOT / f"{arm}-wandb.json.gz"))
    final = max(archive["history"], key=lambda x: x["epoch"])
    rollouts[arm] = final
    for i, depth in enumerate(DEPTHS):
        for split in ["train", "val"]:
            key = f"{split}/loss/depth/depth_{i}_loss"
            rows.append(
                {
                    "arm": arm,
                    "epoch": 70,
                    "depth_m": depth,
                    "variable": "logged_depth_aggregate",
                    "metric": key,
                    "value": float(s[key]),
                }
            )
            for var in VARS:
                key = f"{split}/loss/channel/{var}_{i}_loss"
                rows.append(
                    {
                        "arm": arm,
                        "epoch": 70,
                        "depth_m": depth,
                        "variable": var,
                        "metric": key,
                        "value": float(s[key]),
                    }
                )
        for var in VARS:
            key = f"rollout_val/360d/weighted_rmse/{var}_{i}"
            rows.append(
                {
                    "arm": arm,
                    "epoch": 70,
                    "depth_m": depth,
                    "variable": var,
                    "metric": key,
                    "value": float(final[key]),
                }
            )
(ROOT / "depth-loss-summary.json").write_text(json.dumps(raw, indent=2) + "\n")
with (ROOT / "depth-loss.csv").open("w") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
for row, split in enumerate(["train", "val"]):
    for col, arms in enumerate([["legacy_mse", "wet_mse"], ["oct24", "dynamic_wet"]]):
        ax = axes[row, col]
        for arm in arms:
            vals = [raw[arm][f"{split}/loss/depth/depth_{i}_loss"] for i in range(19)]
            ax.plot(DEPTHS, vals, marker=".", label=LABELS[arm], color=COLORS[arm])
        ax.set(
            xscale="log",
            yscale="log",
            ylabel=f"{split.capitalize()} logged objective",
            title="Static objectives"
            if col == 0
            else "Dynamically weighted objectives",
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
for ax in axes[-1]:
    ax.set_xlabel("Depth (m; logarithmic scale)")
fig.suptitle(
    "Epoch 70: logged objective by depth\nFirst output slot only; surface includes SSH; loss scales differ between objectives",
    fontsize=12,
)
fig.tight_layout()
fig.savefig(ROOT / "depth-objective.png", dpi=160)
plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
for ax, (var, (name, unit)) in zip(axes.flat, VARS.items()):
    for arm in ARMS:
        vals = [
            rollouts[arm][f"rollout_val/360d/weighted_rmse/{var}_{i}"]
            for i in range(19)
        ]
        ax.plot(DEPTHS, vals, marker=".", label=LABELS[arm], color=COLORS[arm])
    ax.set(xscale="log", yscale="log", ylabel=f"RMSE ({unit})", title=name)
    ax.grid(alpha=0.2)
axes[0, 0].legend(fontsize=8)
for ax in axes[-1]:
    ax.set_xlabel("Depth (m; logarithmic scale)")
fig.suptitle(
    "Epoch-70 EMA: fixed 360-day rollout error by depth\nArea-weighted wet-cell RMSE, averaged over rollout times; lower is better",
    fontsize=12,
)
fig.tight_layout()
fig.savefig(ROOT / "depth-rollout.png", dpi=160)
plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
changes = []
for ax, (var, (name, unit)) in zip(axes.flat, VARS.items()):
    for arm, baseline, label in [
        ("wet_mse", "legacy_mse", "Static: wet vs legacy"),
        ("dynamic_wet", "oct24", "Dynamic: wet vs legacy"),
    ]:
        v = np.array(
            [
                rollouts[arm][f"rollout_val/360d/weighted_rmse/{var}_{i}"]
                for i in range(19)
            ]
        )
        b = np.array(
            [
                rollouts[baseline][f"rollout_val/360d/weighted_rmse/{var}_{i}"]
                for i in range(19)
            ]
        )
        delta = 100 * (v / b - 1)
        ax.plot(DEPTHS, delta, marker=".", label=label, color=COLORS[arm])
        for depth, a, base, diff in zip(DEPTHS, v, b, delta):
            changes.append(
                {
                    "comparison": label,
                    "variable": var,
                    "depth_m": depth,
                    "wet_rmse": a,
                    "legacy_rmse": base,
                    "change_percent": diff,
                }
            )
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.set(xscale="log", ylabel="RMSE change (%)", title=name)
    ax.grid(alpha=0.2)
axes[0, 0].legend(fontsize=8)
for ax in axes[-1]:
    ax.set_xlabel("Depth (m; logarithmic scale)")
fig.suptitle(
    "Wet-cell versus legacy denominator: epoch-70 EMA rollout RMSE\nNegative = lower error; positive = higher error; single seed",
    fontsize=12,
)
fig.tight_layout()
fig.savefig(ROOT / "depth-change.png", dpi=160)
plt.close(fig)
with (ROOT / "depth-change.csv").open("w") as f:
    w = csv.DictWriter(f, fieldnames=changes[0].keys())
    w.writeheader()
    w.writerows(changes)
for var in VARS:
    print(
        var,
        [
            (r["comparison"], r["depth_m"], round(r["change_percent"], 2))
            for r in changes
            if r["variable"] == var and r["depth_m"] in [2.5, 550, 1850, 4000, 6000]
        ],
    )
