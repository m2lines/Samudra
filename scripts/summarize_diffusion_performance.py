# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Summarize recorded real-sample timings and first-update gradient checks."""

import argparse
import gzip
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    records = []
    raw = []
    for path in sorted(a.root.glob("job-*/timings.json")):
        data = json.loads(path.read_text())
        raw.append(data)
        base_path = path.parent / "baseline-gradient-sample.npz"
        baseline = dict(np.load(base_path)) if base_path.exists() else None
        baseline_rows = [r for r in data["variants"]["baseline"] if not r["warmup"]]
        if not baseline_rows:
            continue
        baseline_seconds = float(np.median([r["total_seconds"] for r in baseline_rows]))
        for name, updates in data["variants"].items():
            measured = [r for r in updates if not r["warmup"]]
            if len(measured) < 3:
                continue
            median = float(np.median([r["total_seconds"] for r in measured]))
            row = dict(
                job=data["job"],
                gpu=data["gpu"],
                precision=data.get("precision", "bf16"),
                variant=name,
                median_seconds=median,
                min_seconds=min(r["total_seconds"] for r in measured),
                max_seconds=max(r["total_seconds"] for r in measured),
                in_job_speedup=baseline_seconds / median,
                first_update_seconds=updates[0]["total_seconds"],
                first_loss=updates[0]["loss"],
                first_gradient_norm=updates[0]["gradient_norm"],
                peak_allocated_gib=max(r["peak_allocated_gib"] for r in measured),
                job_completed=(path.parent / "COMPLETE.json").exists(),
            )
            candidate = path.parent / f"{name}-gradient-sample.npz"
            if baseline is not None and candidate.exists():
                variant = dict(np.load(candidate))
                assert set(baseline) == set(variant)
                x = np.concatenate(
                    [baseline[k].astype(np.float64) for k in sorted(baseline)]
                )
                y = np.concatenate(
                    [variant[k].astype(np.float64) for k in sorted(baseline)]
                )
                row["gradient_sample"] = dict(
                    count=x.size,
                    relative_l2=float(np.linalg.norm(y - x) / np.linalg.norm(x)),
                    cosine=float(np.dot(x, y) / np.linalg.norm(x) / np.linalg.norm(y)),
                    max_absolute=float(np.max(np.abs(y - x))),
                    exact=bool(np.array_equal(x, y)),
                )
            records.append(row)
    result = dict(
        scope="Three steady-state full-gradient observation updates after two warmups on one fixed real sample, fresh AdamW, no replay/validation/loader/checkpoint costs. First-update gradient samples are first64elements per parameter, not all model gradients. Timings retained from completed variants of subsequently failed jobs are explicitly marked.",
        records=records,
    )
    (a.output / "performance-summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    with gzip.open(a.output / "raw-timings.json.gz", "wt") as f:
        json.dump(raw, f, indent=2)
    bf16 = [
        r
        for r in records
        if r["precision"] == "bf16"
        and r["variant"]
        in ("baseline", "compiled_decoder", "skip_initial", "compiled_skip_initial")
    ]
    fig, ax = plt.subplots(figsize=(10, max(4, len(bf16) * 0.32)), layout="constrained")
    colors = ["#206cb0" if "H200" in r["gpu"] else "#168445" for r in bf16]
    ax.barh(np.arange(len(bf16)), [r["median_seconds"] for r in bf16], color=colors)
    ax.set_yticks(
        np.arange(len(bf16)),
        [
            f"{r['job']}: {r['gpu'].replace('NVIDIA ', '')} / {r['variant']}"
            for r in bf16
        ],
        fontsize=8,
    )
    ax.invert_yaxis()
    ax.set(
        xlabel="Median seconds per optimizer update (lower is faster)",
        title="Real-sample bf16 training benchmarks; excludes replay and validation",
    )
    fig.savefig(a.output / "training-timings.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
