# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Render the eight-arm soft-prior pilot from validated, immutable run receipts."""

import argparse
import csv
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("root", type=Path)
parser.add_argument("controls", type=Path)
args = parser.parse_args()
out = Path(__file__).parent
expected = [0, 1, 50, 100, 200, 500, 750, 1000, 1500, 2000]
decoders = ["control", "twohot", "smoothinit", "both"]
tasks = ["temperature", "all-fields"]
colors = dict(zip(decoders, ["#777777", "#0072b2", "#d55e00", "#009e73"], strict=True))
runs = {}
receipts = {}
reference = None
aggregate, fields = [], []
channel_rows: list[dict] = []
for decoder in decoders:
    for task in tasks:
        arm = f"{task}-{decoder}"
        root = (
            (args.controls / f"{task}-flat8192")
            if decoder == "control"
            else args.root / arm
        )
        complete = json.loads((root / "COMPLETE.json").read_text())
        assert complete["step"] == complete["ema_updates"] == 2000
        assert complete["replicas_agree"]
        assert json.loads((root / "replica-check.json").read_text())["identical"]
        provenance = json.loads((root / "provenance-restart-0.json").read_text())
        cfg = provenance["config"]
        assert cfg["crop"] == 17 and cfg["model"]["width"] == 512
        assert cfg["temperature_loss_only"] == (task == "temperature")
        receipts[arm] = dict(complete=complete, provenance=provenance)
        for kind in ["raw", "ema"]:
            vals = [
                json.loads(p.read_text())
                for p in sorted(root.glob(f"validation-{kind}-*.json"))
            ]
            assert [v["step"] for v in vals] == expected
            runs[task, decoder, kind] = vals
            for v in vals:
                if reference is None:
                    reference = v
                assert v["origins"] == reference["origins"]
                assert len(v["per_channel"]) == 77
                assert "probability_mean_full_grid_nmse" not in v
                for r, ref in zip(
                    v["per_channel"], reference["per_channel"], strict=True
                ):
                    assert r["channel"] == ref["channel"] and r["count"] == ref["count"]
                    np.testing.assert_allclose(
                        r["persistence_full_grid_nmse"],
                        ref["persistence_full_grid_nmse"],
                        rtol=1e-12,
                        atol=1e-15,
                    )
                    assert all(
                        np.isfinite(x)
                        for x in r.values()
                        if isinstance(x, (float, int))
                    )
                base = dict(task=task, decoder=decoder, weights=kind, step=v["step"])
                trained = [
                    r
                    for r in v["per_channel"]
                    if task == "all-fields" or r["channel"].startswith("thetao_")
                ]
                keys = [
                    k
                    for k in trained[0]
                    if k.endswith("nmse")
                    or k
                    in (
                        "full_grid_nll",
                        "wet_cell_nll",
                        "wet_cell_tail_fraction",
                        "wet_cell_leaf_error_rate",
                        "wet_cell_coarse_error_rate",
                        "wet_cell_teacher_coarse_fine_error_rate",
                    )
                ]
                aggregate.append(
                    base | {k: float(np.mean([r[k] for r in trained])) for k in keys}
                )
                channel_rows.extend(
                    base
                    | dict(
                        supervised=task == "all-fields"
                        or r["channel"].startswith("thetao_")
                    )
                    | r
                    for r in v["per_channel"]
                )
                for field in ["thetao", "so", "uo", "vo", "zos"]:
                    rows = [
                        r
                        for r in v["per_channel"]
                        if r["channel"].split("_")[0] == field
                    ]
                    fields.append(
                        base
                        | dict(
                            field=field,
                            supervised=task == "all-fields" or field == "thetao",
                        )
                        | {k: float(np.mean([r[k] for r in rows])) for k in keys}
                    )


assert reference is not None


def write_csv(name, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (
        gzip.open(out / name, "wt", newline="")
        if name.endswith(".gz")
        else (out / name).open("w", newline="") as f
    ):
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


write_csv("validation-aggregate.csv", aggregate)
write_csv("validation-fields.csv", fields)
write_csv("validation-channels.csv.gz", channel_rows)
receipts["support"] = json.loads((args.controls / "configs/support.json").read_text())
receipts["accounting"] = (args.root / "accounting.psv").read_text()
receipts["qualification"] = json.loads((args.root / "QUALIFIED.json").read_text())
with gzip.open(out / "receipts.json.gz", "wt") as f:
    json.dump(receipts, f)


def plot_settings(ax, ylabel, log_y=True):
    ax.set_xlabel("Optimizer update")
    ax.set_ylabel(ylabel)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(left=0)
    if log_y:
        ax.set_yscale("log")
    ax.grid(alpha=0.2)


for early in [False, True]:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), layout="constrained")
    for task, ax in zip(tasks, axes, strict=True):
        for decoder in decoders:
            for kind in ["raw", "ema"]:
                vals = runs[task, decoder, kind]
                vals = [v for v in vals if not early or v["step"] <= 1000]
                ax.plot(
                    [v["step"] for v in vals],
                    [v["temperature_full_grid_nmse"] for v in vals],
                    marker=".",
                    color=colors[decoder],
                    ls="-" if kind == "ema" else "--",
                    label=f"{decoder} {kind}",
                )
        ref_rows = [
            r for r in reference["per_channel"] if r["channel"].startswith("thetao_")
        ]
        baseline = np.mean([r["persistence_full_grid_nmse"] for r in ref_rows])
        ax.axhline(baseline, color="black", ls=":", label="Persistence")
        plot_settings(ax, "Temperature normalized full-grid MSE")
        ax.set_title(f"{task} supervision")
        ax.legend(fontsize=7)
    fig.savefig(
        out / ("temperature-early.png" if early else "temperature-curves.png"), dpi=150
    )
    plt.close(fig)

fig, axes = plt.subplots(2, 4, figsize=(18, 8), layout="constrained")
for i, decoder in enumerate(decoders):
    for j, task in enumerate(tasks):
        ax = axes[j, i]
        for kind in ["raw", "ema"]:
            vals = runs[task, decoder, kind]
            ax.plot(
                [v["step"] for v in vals],
                [v["temperature_full_grid_nll"] for v in vals],
                ".-",
                label=kind,
            )
        plot_settings(ax, "Temperature full-grid NLL (nats)", False)
        ax.set_title(f"{task}: {decoder}")
        ax.legend()
fig.suptitle(
    "Hard-label NLL at fixed 8192 bins; two-hot training changes the target distribution"
)
fig.savefig(out / "temperature-likelihood.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(4, 5, figsize=(18, 12), layout="constrained")
for depth, ax in enumerate(axes.flat):
    if depth == 19:
        ax.axis("off")
        continue
    for task in tasks:
        for decoder in decoders:
            vals = runs[task, decoder, "ema"]
            rows = [
                next(r for r in v["per_channel"] if r["channel"] == f"thetao_{depth}")
                for v in vals
            ]
            ax.plot(
                [v["step"] for v in vals],
                [r["wet_cell_rmse"] for r in rows],
                color=colors[decoder],
                ls="-" if task == "temperature" else "--",
                label=f"{task} {decoder}",
            )
    persistence_rmse = rows[0]["wet_cell_rmse"] * np.sqrt(
        rows[0]["persistence_full_grid_nmse"] / rows[0]["full_grid_nmse"]
    )
    ax.axhline(persistence_rmse, color="black", ls=":", label="Persistence")
    plot_settings(ax, "Wet-cell RMSE (°C)")
    ax.set_title(f"{rows[0]['depth_m']:g} m")
axes.flat[0].legend(fontsize=6)
fig.savefig(out / "temperature-depth-rmse.png", dpi=140)
plt.close(fig)

fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
for ax, field in zip(
    axes.flat, ["thetao", "so", "uo", "vo", "zos", "all"], strict=True
):
    for decoder in decoders:
        for kind in ["raw", "ema"]:
            selected = [
                r
                for r in (aggregate if field == "all" else fields)
                if r["task"] == "all-fields"
                and r["decoder"] == decoder
                and r["weights"] == kind
                and (field == "all" or r["field"] == field)
            ]
            ax.plot(
                [r["step"] for r in selected],
                [r["full_grid_nmse"] for r in selected],
                color=colors[decoder],
                ls="-" if kind == "ema" else "--",
                label=f"{decoder} {kind}",
            )
    ax.axhline(
        selected[0]["persistence_full_grid_nmse"],
        color="black",
        ls=":",
        label="Persistence",
    )
    plot_settings(ax, "Normalized full-grid MSE")
    ax.set_title(field)
axes.flat[0].legend(fontsize=7)
fig.savefig(out / "all-fields-curves.png", dpi=150)
plt.close(fig)

endpoints = [
    r for r in fields if r["step"] == 2000 and r["weights"] == "ema" and r["supervised"]
]
write_csv("endpoints.csv", endpoints)


crossings = []
cases = []
initializations = []
for decoder in decoders:
    for task in tasks:
        root = (
            args.controls / f"{task}-flat8192"
            if decoder == "control"
            else args.root / f"{task}-{decoder}"
        )
        logs = [
            json.loads(line)
            for line in (root / "training.jsonl").read_text().splitlines()
        ]
        times = [(0, 0.0)] + [
            (r["step"], r["elapsed_seconds"]) for r in logs if "elapsed_seconds" in r
        ]
        for kind in ("raw", "ema"):
            vals = runs[task, decoder, kind]
            for threshold in [0.1, 0.02, 0.01, 0.005, 0.001, baseline]:
                hit = next(
                    (v for v in vals if v["temperature_full_grid_nmse"] <= threshold),
                    None,
                )
                step = None if hit is None else hit["step"]
                lower = None if hit is None else max(t for s, t in times if s <= step)
                upper = None if hit is None else min(t for s, t in times if s >= step)
                crossings.append(
                    dict(
                        task=task,
                        decoder=decoder,
                        weights=kind,
                        threshold=threshold,
                        threshold_label="persistence"
                        if threshold == baseline
                        else str(threshold),
                        first_observed_update=step,
                        elapsed_lower_seconds=lower,
                        elapsed_upper_seconds=upper,
                        retained_at_2000=vals[-1]["temperature_full_grid_nmse"]
                        <= threshold,
                    )
                )
            final = vals[-1]
            if "depth13_cases" in final:
                rows = final["depth13_cases"]
                total = sum(r["weighted_squared_error"] for r in rows)
                expected_row = next(
                    r for r in final["per_channel"] if r["channel"] == "thetao_13"
                )
                np.testing.assert_allclose(
                    total / len(rows), expected_row["full_grid_nmse"], rtol=1e-6
                )
                for rank, row in enumerate(
                    sorted(
                        rows, key=lambda r: r["weighted_squared_error"], reverse=True
                    ),
                    1,
                ):
                    cases.append(
                        dict(
                            task=task,
                            decoder=decoder,
                            weights=kind,
                            error_rank=rank,
                            share_of_depth13_error=row["weighted_squared_error"]
                            / total,
                        )
                        | row
                    )
write_csv("time-to-skill.csv", crossings)
with gzip.open(out.parent / "discrete-rarity/validation-cases.csv.gz", "rt") as f:
    prior_cases = list(csv.DictReader(f))
metadata = {(int(r["origin"]), int(r["row"]), int(r["col"])): r for r in prior_cases}
source = receipts["temperature-twohot"]["provenance"]["source"]
scale, mean = source["std"][51], source["mean"][51]
for case in cases:
    prior = metadata[case["origin"], case["row"], case["col"]]
    case.update({k: prior[k] for k in ("lat", "lon", "origin_date", "target_date")})
    for field in ("prediction", "target", "current"):
        case[field + "_c"] = case[field + "_normalized"] * scale + mean
    np.testing.assert_allclose(case["target_c"], float(prior["target_c"]), atol=1e-6)
write_csv("depth13-cases.csv", cases)


fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), layout="constrained")
for task, ax in zip(tasks, axes, strict=True):
    for decoder in decoders:
        root = (
            args.controls / f"{task}-flat8192"
            if decoder == "control"
            else args.root / f"{task}-{decoder}"
        )
        logs = [
            json.loads(line)
            for line in (root / "training.jsonl").read_text().splitlines()
        ]
        times = [(0, 0.0)] + [
            (r["step"], r["elapsed_seconds"]) for r in logs if "elapsed_seconds" in r
        ]
        vals = runs[task, decoder, "ema"]
        xlo = [max(t for s, t in times if s <= v["step"]) / 60 for v in vals]
        xhi = [min(t for s, t in times if s >= v["step"]) / 60 for v in vals]
        mid = (np.array(xlo) + np.array(xhi)) / 2
        ax.errorbar(
            mid,
            [v["temperature_full_grid_nmse"] for v in vals],
            xerr=(np.array(xhi) - np.array(xlo)) / 2,
            color=colors[decoder],
            label=decoder,
            marker=".",
        )
    ax.axhline(baseline, color="black", ls=":", label="Persistence")
    ax.set_yscale("log")
    ax.set_xlabel("Elapsed training-loop minutes (includes validation)")
    ax.set_ylabel("Temperature normalized full-grid MSE, EMA")
    ax.set_title(f"{task} supervision")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
fig.savefig(out / "temperature-time.png", dpi=150)
plt.close(fig)

for decoder in decoders:
    for task in tasks:
        root = args.root / f"qual-v2-{task}-{decoder}"
        p = json.loads((root / "provenance-restart-0.json").read_text())
        v = json.loads((root / "validation-raw-0000000.json").read_text())
        initializations.append(
            dict(
                task=task,
                decoder=decoder,
                **p["initial_head_statistics"]["used"],
                validation_logit_rms=v["initial_validation_logits_rms_all_query_slots"],
            )
        )
write_csv("initialization.csv", initializations)
