# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Render matched classifier and regression validation receipts."""

import argparse
import csv
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("classifier", type=Path)
p.add_argument("regression", type=Path)
a = p.parse_args()
out = Path(__file__).parent
runs = {}
for model, root in [("classification", a.classifier), ("regression", a.regression)]:
    for kind in ["raw", "ema"]:
        runs[model, kind] = [
            json.loads(p.read_text())
            for p in sorted(root.glob(f"validation-{kind}-*.json"))
            if int(p.stem.split("-")[-1]) <= 15000
        ]
expected = [0, 1, 50, 100, 200, 500, 750] + list(range(1000, 15001, 1000))
complete = json.loads((a.classifier / "COMPLETE.json").read_text())
assert (
    complete["step"] == complete["ema_updates"] == 15000 and complete["replicas_agree"]
)
assert json.loads((a.classifier / "replica-check.json").read_text())["identical"]
ref = runs["regression", "ema"][-1]
for (model, kind), vals in runs.items():
    assert [v["step"] for v in vals] == (
        expected
        if model == "classification"
        else [0, 1] + list(range(1000, 15001, 1000))
    )
    for v in vals:
        assert v["origins"] == ref["origins"]
        for row, r in zip(v["per_channel"], ref["per_channel"], strict=True):
            assert row["channel"] == r["channel"] and row["count"] == r["count"]
            np.testing.assert_allclose(
                row["persistence_full_grid_nmse"],
                r["persistence_full_grid_nmse"],
                rtol=1e-12,
                atol=1e-15,
            )

agg = []
channel: list[dict] = []
fields = []
for (model, kind), vals in runs.items():
    for v in vals:
        base = dict(model=model, weights=kind, step=v["step"])
        agg.append(base | {k: x for k, x in v.items() if isinstance(x, (int, float))})
        channel.extend(base | r for r in v["per_channel"])
        for field in ["thetao", "so", "uo", "vo", "zos"]:
            rows = [r for r in v["per_channel"] if r["channel"].split("_")[0] == field]
            fields.append(
                base
                | dict(field=field)
                | {
                    k: float(np.mean([r[k] for r in rows]))
                    for k in rows[0]
                    if k.endswith("nmse")
                    or k in ["full_grid_nll", "wet_cell_nll", "wet_cell_tail_fraction"]
                }
            )


def csv_write(name, rows, compressed=False):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (
        gzip.open(out / name, "wt", newline="")
        if compressed
        else (out / name).open("w", newline="")
    ) as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


csv_write("validation-aggregate.csv", agg)
csv_write("validation-fields.csv", fields)
csv_write("validation-channels.csv.gz", channel, True)
for early in [False, True]:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    for ax, field in zip(
        axes.flat, ["thetao", "so", "uo", "vo", "zos", "all"], strict=True
    ):
        for (model, kind), vals in runs.items():
            vals = [v for v in vals if not early or v["step"] <= 1000]
            x = [v["step"] for v in vals]
            color = "#0072b2" if model == "classification" else "#d55e00"
            for mean in [False, True] if model == "classification" else [False]:
                y = []
                for v in vals:
                    rows = [
                        r
                        for r in v["per_channel"]
                        if field == "all" or r["channel"].split("_")[0] == field
                    ]
                    key = (
                        "probability_mean_full_grid_nmse" if mean else "full_grid_nmse"
                    )
                    y.append(np.mean([r[key] for r in rows]))
                label = f"{model} {kind}" + (
                    " mean" if mean else " argmax" if model == "classification" else ""
                )
                if model == "regression":
                    x = x[:2] + [float("nan")] + x[2:]
                    y = y[:2] + [float("nan")] + y[2:]
                ax.plot(
                    x,
                    y,
                    marker=".",
                    color=("#009e73" if mean else color),
                    ls="-" if kind == "ema" else "--",
                    label=label,
                )
        ax.set(
            title=field,
            xlabel="Update",
            ylabel="Normalized full-grid MSE",
            yscale="log",
        )
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlim(left=0)
        ax.grid(alpha=0.2)
    axes.flat[-1].legend(fontsize=7)
    fig.savefig(out / ("early-mse.png" if early else "full-mse.png"), dpi=150)
    plt.close(fig)
fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
for ax, field in zip(
    axes.flat, ["thetao", "so", "uo", "vo", "zos", "all"], strict=True
):
    for kind in ["raw", "ema"]:
        rows = [
            r
            for r in (fields if field != "all" else agg)
            if r["model"] == "classification"
            and r["weights"] == kind
            and (field == "all" or r["field"] == field)
        ]
        ax.plot(
            [r["step"] for r in rows],
            [r["full_grid_nll"] for r in rows],
            ".-",
            label=kind,
        )
    ax.set(title=field, xlabel="Update", ylabel="Full-grid categorical NLL (nats)")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(left=0)
    ax.grid(alpha=0.2)
    ax.legend()
fig.savefig(out / "likelihood.png", dpi=150)
plt.close(fig)
fig, axes = plt.subplots(4, 5, figsize=(17, 12), layout="constrained")
for depth, ax in enumerate(axes.flat):
    if depth == 19:
        ax.axis("off")
        continue
    for (model, kind), vals in runs.items():
        r = [
            next(r for r in v["per_channel"] if r["channel"] == f"thetao_{depth}")
            for v in vals
        ]
        x = [v["step"] for v in vals]
        y = [row["wet_cell_rmse"] for row in r]
        if model == "regression":
            x = x[:2] + [float("nan")] + x[2:]
            y = y[:2] + [float("nan")] + y[2:]
        ax.plot(
            x,
            y,
            marker=".",
            label=f"{model} {kind}",
            ls="-" if kind == "ema" else "--",
        )
    ax.set(
        title=f"T at {r[0]['depth_m']:g} m",
        xlabel="Update",
        ylabel="Wet-cell RMSE (°C)",
        yscale="log",
    )
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(left=0)
    ax.grid(alpha=0.2)
axes.flat[0].legend(fontsize=7)
fig.savefig(out / "temperature-depth-rmse.png", dpi=140)
plt.close(fig)
fig, axes = plt.subplots(4, 5, figsize=(17, 12), layout="constrained")
for depth, ax in enumerate(axes.flat):
    if depth == 19:
        ax.axis("off")
        continue
    for kind in ["raw", "ema"]:
        vals = runs["classification", kind]
        rows = [
            next(r for r in v["per_channel"] if r["channel"] == f"thetao_{depth}")
            for v in vals
        ]
        ax.plot(
            [v["step"] for v in vals],
            [r["wet_cell_nll"] for r in rows],
            ".-",
            label=kind,
        )
    ax.set(
        title=f"T at {rows[0]['depth_m']:g} m",
        xlabel="Update",
        ylabel="Wet-cell NLL (nats)",
    )
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(left=0)
    ax.grid(alpha=0.2)
axes.flat[0].legend()
fig.savefig(out / "temperature-depth-nll.png", dpi=140)
plt.close(fig)
print(json.dumps([r for r in fields if r["step"] == 15000], indent=2))
