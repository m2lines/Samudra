#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Reconcile saved fields, paired dates, and physical diagnostic metrics."""

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from samudra.experiments.initializer_diagnostics import masked_box
from samudra.experiments.surface_adaptation_analysis import read_bytes

RUNS = ["precision", "continued-control-eval", "field-specialization-eval"]


def analyze(root, output, origins):
    output.mkdir(parents=True, exist_ok=True)
    expected = json.loads(origins.read_text())
    metadata_reference = {}
    truth_reference = {}
    frames = {}
    audit = {}
    scales = []
    for run in RUNS:
        table = pd.read_csv(root / run / "heldout-bf16.csv").sort_values("index")
        assert table.date.tolist() == expected
        assert len(table) == 99 and table["index"].is_unique
        frames[run] = table
        with np.load(
            io.BytesIO(read_bytes(root / run / "heldout-bf16-maps.npz"))
        ) as data:
            mask = data["mask"]
            lat = data["latitude"]
            std = float(data["std"])
            for key in ["mask", "latitude", "longitude", "std", "mean"]:
                if run == RUNS[0]:
                    metadata_reference[key] = data[key].copy()
                else:
                    np.testing.assert_array_equal(data[key], metadata_reference[key])
            weight = mask * np.cos(np.deg2rad(lat))[:, None]
            weight = weight / weight.sum()

            def avg(v):
                return float(np.sum(v.astype(np.float64) * weight))

            moments = {
                n: np.zeros(4)
                for n in ["raw", "lowpass3", "lowpass9", "highpass3", "highpass9"]
            }
            for row in table.to_dict("records"):
                idx = str(row["index"])
                t, p, c = [
                    data[idx + "_" + n] for n in ["truth", "prediction", "climatology"]
                ]
                assert all(
                    np.isfinite(x).all() and (x[~mask] == 0).all() for x in [t, p, c]
                )
                assert str(data[idx + "_date"]) == row["date"]
                np.testing.assert_allclose(
                    avg((p - t) ** 2), row["current_field_mse"], rtol=2e-6, atol=1e-9
                )
                np.testing.assert_allclose(
                    avg((t - c) ** 2), row["climatology_mse"], rtol=2e-6, atol=1e-9
                )
                if run == RUNS[0]:
                    truth_reference[idx] = (t.copy(), c.copy())
                else:
                    np.testing.assert_array_equal(t, truth_reference[idx][0])
                    np.testing.assert_array_equal(c, truth_reference[idx][1])
                pa, ta = p - c, t - c
                components = {"raw": (pa, ta)}
                for size in [3, 9]:
                    pair = torch.from_numpy(np.stack([pa, ta]))[:, None]
                    low = masked_box(pair, torch.from_numpy(mask)[None], size).numpy()[
                        :, 0
                    ]
                    components[f"lowpass{size}"] = (low[0], low[1])
                    components[f"highpass{size}"] = (pa - low[0], ta - low[1])
                for name, (a, b) in components.items():
                    moments[name] += np.array(
                        [avg(a * a), avg(b * b), avg(a * b), avg((a - b) ** 2)]
                    )
            for name, m in moments.items():
                m = m / len(table)
                scales.append(
                    dict(
                        run=run,
                        scale=name,
                        physical_rmse=np.sqrt(m[3]) * std,
                        anomaly_rms_ratio=np.sqrt(m[0] / m[1]),
                        anomaly_cosine=m[2] / np.sqrt(m[0] * m[1]),
                    )
                )
            audit[run] = {
                "origins": len(table),
                "maps_match_metrics": True,
                "finite": True,
                "std": std,
            }
    pd.DataFrame(scales).to_csv(output / "spatial-scales.csv", index=False)
    paired = []
    rng = np.random.default_rng(1729)
    base = frames[RUNS[0]]
    years = pd.to_datetime(base.date).dt.year.to_numpy()
    unique = np.unique(years)
    for run in RUNS[1:]:
        candidate = frames[run]
        b = base.current_field_mse.to_numpy()
        c = candidate.current_field_mse.to_numpy()
        estimates = []
        for _ in range(3000):
            sample_indices = np.concatenate(
                [
                    np.flatnonzero(years == y)
                    for y in rng.choice(unique, len(unique), replace=True)
                ]
            )
            estimates.append(
                1 - np.sqrt(c[sample_indices].mean() / b[sample_indices].mean())
            )
        low, high = np.quantile(estimates, [0.025, 0.975])
        paired.append(
            dict(
                run=run,
                rmse_reduction=1 - np.sqrt(c.mean() / b.mean()),
                year_block_low=low,
                year_block_high=high,
            )
        )
    pd.DataFrame(paired).to_csv(output / "paired-vs-D.csv", index=False)
    yearly = []
    for run, table in frames.items():
        for year, g in table.groupby(pd.to_datetime(table.date).dt.year):
            yearly.append(
                dict(
                    run=run,
                    year=year,
                    origins=len(g),
                    physical_rmse=np.sqrt(g.current_field_mse.mean())
                    * audit[run]["std"],
                    mse_skill=1 - g.current_field_mse.mean() / g.climatology_mse.mean(),
                )
            )
    pd.DataFrame(yearly).to_csv(output / "yearly.csv", index=False)
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    print(pd.DataFrame(paired).to_string(index=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--origins",
        type=Path,
        default=Path("docs/experiments/surface-wave3-origins.json"),
    )
    args = p.parse_args()
    torch.set_num_threads(1)
    analyze(args.raw, args.output, args.origins)


if __name__ == "__main__":
    main()
