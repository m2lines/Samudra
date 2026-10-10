# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import csv
import json
import math
from pathlib import Path

root = Path("experiments/current-stochastic-depth-1deg")
res = root / "torch-seeds/results"
obs = {}
frames = []
for seed in (15, 16, 17):
    for variant in ("control", "sd01"):
        p = (
            root / "observations/observation_metrics.csv"
            if seed == 15
            else res / f"{variant}-seed{seed}/observation_metrics.csv"
        )
        rows = list(csv.DictReader(p.open()))
        assert all(math.isfinite(float(r["value"])) for r in rows)
        prefix = f"{variant}_" if seed == 15 else f"{variant}_seed{seed}_"
        for r in rows:
            if not r["model"].startswith(prefix):
                continue
            r = dict(
                r, seed=seed, variant=variant, checkpoint=r["model"][len(prefix) :]
            )
            frames.append(r)
            if r["period_kind"] == "primary_complete_years":
                obs[(seed, variant, r["checkpoint"], r["metric"], r["depth"])] = float(
                    r["value"]
                )
assert len(frames) == 612
with (res / "observations-all-seeds.csv").open("w") as f:
    w = csv.DictWriter(f, fieldnames=list(frames[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(frames)
comparisons = []
for (seed, v, cp, metric, depth), c in obs.items():
    if v != "control":
        continue
    sd = obs[(seed, "sd01", cp, metric, depth)]
    comparisons.append(
        dict(
            seed=seed,
            checkpoint=cp,
            metric=metric,
            depth=depth,
            control=c,
            sd01=sd,
            change_percent=100 * (sd / c - 1),
        )
    )
assert len(comparisons) == 30
with (res / "observation-paired-comparison.csv").open("w") as f:
    w = csv.DictWriter(f, fieldnames=list(comparisons[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(comparisons)
evals = {}
trains = {}
ec = []
for seed in (15, 16, 17):
    for v in ("control", "sd01"):
        p = (
            root / f"results/{v}-eval-summary.json"
            if seed == 15
            else res / f"{v}-seed{seed}/eval-summary.json"
        )
        evals[seed, v] = {x["label"]: x["metrics"] for x in json.loads(p.read_text())}
        p = (
            root / f"results/{v}-training-scalars.json"
            if seed == 15
            else res / f"{v}-seed{seed}/training-scalars.json"
        )
        trains[seed, v] = json.loads(p.read_text())
    for cp in ("epoch_0070", "ema_latest"):
        control_metrics = evals[seed, "control"][cp]
        sd_metrics = evals[seed, "sd01"][cp]
        assert control_metrics.keys() == sd_metrics.keys()
        assert all(
            math.isfinite(x) for x in [*control_metrics.values(), *sd_metrics.values()]
        )
        for k in control_metrics:
            ec.append(
                dict(
                    seed=seed,
                    checkpoint=cp,
                    metric=k,
                    control=control_metrics[k],
                    sd01=sd_metrics[k],
                    change_percent=100 * (sd_metrics[k] / control_metrics[k] - 1)
                    if control_metrics[k] != 0
                    else "",
                )
            )
with (res / "eval-paired-comparison.csv").open("w") as f:
    w = csv.DictWriter(f, fieldnames=list(ec[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(ec)
print("OBS PAIRED")
for r in comparisons:
    print(
        r["seed"],
        r["checkpoint"],
        r["metric"],
        r["depth"],
        f"{r['control']:.6g} {r['sd01']:.6g} {r['change_percent']:+.2f}%",
    )
print("TRAIN + TIME MEAN")
for seed in (15, 16, 17):
    for k in ("val/mean/loss", "rollout_val/360d/normalized_rmse/channel_mean"):
        c = trains[seed, "control"][k]
        d = trains[seed, "sd01"][k]
        print(seed, k, c, d, 100 * (d / c - 1))
    for cp in ("epoch_0070", "ema_latest"):
        k = "inference/time_mean_norm/rmse/channel_mean"
        c = evals[seed, "control"][cp][k]
        d = evals[seed, "sd01"][cp][k]
        print(seed, cp, c, d, 100 * (d / c - 1))
print(
    "WORSE",
    sum(float(r["change_percent"]) > 0 for r in comparisons),
    "of",
    len(comparisons),
)
