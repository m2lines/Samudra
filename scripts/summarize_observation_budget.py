#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Summarize an audited observation-budget snapshot without selecting on held-out data."""

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from samudra.experiments.observation_metrics import selection_score

p = argparse.ArgumentParser()
p.add_argument("snapshot", type=Path)
p.add_argument("output", type=Path)
args = p.parse_args()
s = json.loads(args.snapshot.read_text())
out = args.output
out.mkdir(parents=True, exist_ok=True)
labels = {
    "obs0": "D → observations",
    "obs25": "D → 25% OM4 → observations",
    "obs50": "D → 50% OM4 → observations",
    "obs75": "D → 75% OM4 → observations",
    "random-4000": "Observation-only random (4k)",
    "random": "Observation-only random (8k)",
}
ref = s["reference"]
assert len(ref["spectral_keys"]) == 27
expected = [f"{y}-{m:02d}" for y in range(2015, 2023) for m in range(1, 13)]
methods = [
    "selected",
    "selected-inferred-persistence",
    "selected-inferred-anomaly-persistence",
    "source-with-zero-forcing",
    "source-inferred-persistence",
    "inferred-anomaly-persistence",
    "seasonal-climatology",
]
rows = []
selections = {}
annual = {}
curves = []
for arm, label in labels.items():
    d = s["arms"][arm]
    e = d["evaluation"]
    for ev in d["validation_events"]:
        curves.append(
            {
                "arm": arm,
                "label": label,
                "phase": ev["phase"],
                "step": ev["step"],
                "score": ev["validation/obs_score"],
                "best_score": ev["validation/best_obs_score"],
                "time_utc": ev["time_utc"],
            }
        )
    if not e.get("COMPLETE"):
        continue
    b = d["best"]
    assert d["TRAIN_COMPLETE"]
    assert e["COMPLETE"]["origins"] == 96
    assert (
        e["selected"]["sha256"]
        == b["checkpoint_sha256"]
        == e["COMPLETE"]["selected_sha256"]
    )
    if "anomaly-year-report" not in e:
        raise ValueError("Missing diagnostics " + arm)
    assert e["anomaly-year-report"]["evaluation_input"] == e["evaluation-input"]
    selections[arm] = {
        "label": label,
        "score": b["score"],
        "phase": b["phase"],
        "step": b["step"],
        "sha256": b["checkpoint_sha256"],
        "completion": d["TRAIN_COMPLETE"],
    }
    annual[arm] = e["anomaly-year-report"]
    for method in methods:
        result = e[method]["metrics"] if method == "selected" else e[method]
        assert result["origins"] == expected
        row = {
            "arm": arm,
            "label": label,
            "method": method,
            "composite": selection_score(result, ref["control"], ref["spectral_keys"]),
            "spectral_error_dex": float(
                np.mean(
                    [result["spectra"][k]["error_dex"] for k in ref["spectral_keys"]]
                )
            ),
        }
        row.update(result["metrics"])
        rows.append(row)
for name, data in [("heldout-metrics", rows), ("validation-curves", curves)]:
    if data:
        with (out / (name + ".csv")).open("w") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
accounting = []
for line in s["accounting"].splitlines()[1:]:
    v = line.split("|")
    if len(v) < 5:
        continue
    n = next(
        (int(t.split("=")[1]) for t in v[3].split(",") if t.startswith("gres/gpu=")), 0
    )
    accounting.append(
        {
            "job": v[0],
            "state": v[1],
            "seconds": int(v[2]),
            "gpu_count": n,
            "gpu_hours": int(v[2]) * n / 3600,
            "exit_code": v[4],
        }
    )
paired = {}
if "obs0" in annual:
    base = annual["obs0"]["methods"]["selected"]["annual"]
    for a, annual_data in annual.items():
        if a == "obs0":
            continue
        other = annual_data["methods"]["selected"]["annual"]
        delta = {
            y: base[y]["composite_fixed_validation_scale"]
            - other[y]["composite_fixed_validation_scale"]
            for y in base
        }
        paired[a] = {
            "obs0_minus_candidate_by_year": delta,
            "years_obs0_better": sum(v < 0 for v in delta.values()),
            "mean": float(np.mean(list(delta.values()))),
        }
summary = {
    "snapshot_utc": s["snapshot_utc"],
    "normalization": "Frozen validation climatology; no held-out checkpoint selection",
    "selections": selections,
    "heldout": rows,
    "paired_year_obs0": paired,
    "gpu_hours": sum(r["gpu_hours"] for r in accounting),
    "accounting": accounting,
}
(out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
(out / "diagnostics.json.gz").write_bytes(
    gzip.compress(
        (json.dumps(annual, indent=2, allow_nan=False) + "\n").encode(), mtime=0
    )
)
(out / "evidence.json.gz").write_bytes(
    gzip.compress(args.snapshot.read_bytes(), mtime=0)
)
header = "<!--\nSPDX-FileCopyrightText: 2026 Samudra Authors\n\nSPDX-License-Identifier: CC-BY-4.0\n-->\n\n"
lines = [
    "# Held-out component results",
    "",
    "All 96 monthly origins in 2015–2022; 5/15/30-day forecast diagnostics. No continuous eight-year rollout. Composite uses the frozen validation climatology denominators; lower is better. OHC is shown in GJ/m² (CSV stores J/m²).",
    "",
    "| Model | Control | Composite | Spectral (dex) | SST (°C) | Velocity (m/s) | EKE (m²/s²) | OHC 0–700 | OHC 700–2000 |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    lines.append(
        f"| {r['label']} | {r['method']} | {r['composite']:.4f} | {r['spectral_error_dex']:.4f} | {r['sst_rmse']:.4f} | {r['velocity_rmse']:.4f} | {r['eke_rmse']:.5f} | {r['ohc_0_700_rmse'] / 1e9:.4f} | {r['ohc_700_2000_rmse'] / 1e9:.4f} |"
    )
lines += [
    "",
    "`selected` is the validation-selected complete model. `selected-inferred-persistence` keeps its inferred state fixed without evolution. `selected-inferred-anomaly-persistence` advances the training seasonal mean of interior T/S while keeping the initial anomaly; surface fields persist. `source-with-zero-forcing` uses that arm’s input OM4 checkpoint with a zero-output forcing adapter: original D for the zero-OM4 and random arms, or the corresponding continued-OM4 prefix for the other arms. `source-inferred-persistence` and `inferred-anomaly-persistence` use that same source checkpoint’s initializer with those persistence rules. `seasonal-climatology` is the training seasonal climatology. All controls are evaluated on identical observations.",
]
(out / "heldout-table.md").write_text(header + "\n".join(lines) + "\n")
# Plot all observed validations, retaining terminal deterioration rather than hiding it.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 5))
for arm, label in labels.items():
    points = [c for c in curves if c["arm"] == arm and c["phase"] == "joint"]
    if not points:
        continue
    # Requeued validation may be replayed; preserve the last record for each update.
    points = list({c["step"]: c for c in points}.values())
    points.sort(key=lambda c: c["step"])
    ax.plot(
        [c["step"] for c in points],
        [c["score"] for c in points],
        marker="o",
        label=label.replace(" (8k)", ""),
    )
ax.set(
    xlabel="Observation joint optimizer updates (batch 8)",
    ylabel="Integrated + spectral validation score ↓",
    title="One seed; common 1,000 observation reconstruction updates",
)
ax.grid(alpha=0.2)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(out / "validation-curves.png", dpi=160)
plt.close(fig)
print(
    json.dumps(
        {
            "gpu_hours": summary["gpu_hours"],
            "completed": list(selections),
            "heldout_selected": [
                {
                    k: r[k]
                    for k in ["arm", "composite", "spectral_error_dex", "sst_rmse"]
                }
                for r in rows
                if r["method"] == "selected"
            ],
        },
        indent=2,
    )
)
