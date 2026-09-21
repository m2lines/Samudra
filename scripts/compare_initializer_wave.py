#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Paired calendar-year bootstrap of fixed-origin initializer comparisons."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def paired_intervals(reference, candidate, draws=4000, seed=1729):
    """Resample whole years, retaining all origins and their original weights."""
    a = reference.set_index("origin_time")["normalized_mse"].sort_index()
    b = candidate.set_index("origin_time")["normalized_mse"].sort_index()
    if not a.index.is_unique or not b.index.is_unique or not a.index.equals(b.index):
        raise ValueError("Comparisons require unique, exactly matched origins")
    values = np.column_stack((a.to_numpy(), b.to_numpy()))
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid squared errors")
    years = pd.to_datetime(a.index).year.to_numpy()
    unique = np.unique(years)
    if len(unique) < 2:
        raise ValueError("Year-block uncertainty requires at least two years")
    totals = np.stack([values[years == year].sum(0) for year in unique])
    counts = np.array([(years == year).sum() for year in unique])
    sample = np.random.default_rng(seed).integers(0, len(unique), (draws, len(unique)))
    boot = np.sqrt(totals[sample].sum(1) / counts[sample].sum(1)[:, None])
    score = np.sqrt(values.mean(0))
    difference = boot[:, 1] - boot[:, 0]
    interval = np.quantile(difference, [0.025, 0.975])
    relative = np.full(draws, np.nan)
    np.divide(boot[:, 0] - boot[:, 1], boot[:, 0], out=relative, where=boot[:, 0] > 0)
    relative *= 100
    relative_interval = (
        np.quantile(relative, [0.025, 0.975])
        if np.isfinite(relative).all()
        else [np.nan, np.nan]
    )
    return {
        "origins": len(a),
        "years": len(unique),
        "bootstrap_draws": draws,
        "reference_rmse": score[0],
        "candidate_rmse": score[1],
        "candidate_minus_reference_rmse": score[1] - score[0],
        "difference_ci_low": interval[0],
        "difference_ci_high": interval[1],
        "rmse_reduction_percent": 100 * (1 - score[1] / score[0])
        if score[0]
        else np.nan,
        "reduction_ci_low": relative_interval[0],
        "reduction_ci_high": relative_interval[1],
    }


def compare(origins, reference, output, draws=4000):
    frame = pd.read_csv(origins)
    frame = frame[frame["mode"] == "inferred"]
    if reference not in set(frame["run"]):
        raise ValueError("Reference run is absent")
    group_keys = ["region", "lead_days", "variable"]
    baseline = frame[frame["run"] == reference]
    baseline_groups = dict(tuple(baseline.groupby(group_keys)))
    rows = []
    for run, candidate in frame[frame["run"] != reference].groupby("run"):
        groups = dict(tuple(candidate.groupby(group_keys)))
        if groups.keys() != baseline_groups.keys():
            raise ValueError("Candidate and reference metric groups differ")
        for key, values in groups.items():
            rows.append(
                {
                    "reference": reference,
                    "candidate": run,
                    **dict(zip(group_keys, key, strict=True)),
                    **paired_intervals(baseline_groups[key], values, draws),
                }
            )
    if not rows:
        raise ValueError("No candidate comparisons")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origins", required=True)
    parser.add_argument("--reference", default="A")
    parser.add_argument("--output", required=True)
    parser.add_argument("--draws", type=int, default=4000)
    args = parser.parse_args()
    compare(args.origins, args.reference, args.output, args.draws)


if __name__ == "__main__":
    main()
