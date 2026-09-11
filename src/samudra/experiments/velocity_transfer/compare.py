# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Compare matched velocity forecasts with paired seed and calendar-quarter resampling."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_scores(path: Path):
    manifest = json.loads((path / "manifest.json").read_text())
    if not manifest["complete"]:
        raise ValueError(f"Incomplete evaluation: {path}")
    scores = pd.read_csv(path / "scores.csv")
    scores = (
        scores[scores.method == "samudra"]
        .set_index(["anchor", "target", "lead_step", "region"])
        .sort_index()
    )
    if scores.index.has_duplicates:
        raise ValueError(f"Duplicate forecast rows: {path}")
    return manifest, scores


def compare(control_paths, candidate_paths, draws=2000):
    controls = [read_scores(Path(p)) for p in control_paths]
    candidates = [read_scores(Path(p)) for p in candidate_paths]
    control_by_seed = {m["config"]["seed"]: (m, s) for m, s in controls}
    candidate_by_seed = {m["config"]["seed"]: (m, s) for m, s in candidates}
    if len(control_by_seed) != len(controls) or len(candidate_by_seed) != len(
        candidates
    ):
        raise ValueError("Duplicate seed in a comparison arm")
    if set(control_by_seed) != set(candidate_by_seed) or not controls:
        raise ValueError("Both arms must have the same nonempty set of seeds")
    if len({m["config"]["variant"] for m, _ in candidates}) != 1:
        raise ValueError("Candidate runs must use the same variant")
    pairs = []
    reference_index = controls[0][1].index
    for seed in sorted(control_by_seed):
        cm, c = control_by_seed[seed]
        mm, m = candidate_by_seed[seed]
        if cm["config"]["variant"] != "D0" or cm["split"] != mm["split"]:
            raise ValueError("Require a D0 control and matching evaluation splits")
        for key in ("gpu_hours", "widths", "data_root"):
            if cm["config"][key] != mm["config"][key]:
                raise ValueError(f"Unmatched training configuration: {key}")
        if not c.index.equals(m.index) or not c.index.equals(reference_index):
            raise ValueError("Unmatched forecast dates, targets, regions, or leads")
        for column in ("weight", "valid_cell_count", "actual_lead_days"):
            if not np.allclose(c[column], m[column], rtol=1e-7, atol=1e-8):
                raise ValueError(f"Unmatched evaluation cohort: {column}")
        rows = (
            c[["weighted_squared_error", "weight"]]
            .rename(columns={"weighted_squared_error": "control_sse"})
            .copy()
        )
        rows["candidate_sse"] = m.weighted_squared_error
        rows = rows.reset_index()
        rows["block"] = pd.to_datetime(rows.anchor).dt.to_period("Q").astype(str)
        rows["seed"] = seed
        pairs.append(rows)
    data = pd.concat(pairs, ignore_index=True)
    rng = np.random.default_rng(2026)
    result = []
    for (region, step), group in data.groupby(["region", "lead_step"]):
        grouped = group.groupby(["seed", "block"])[
            ["control_sse", "candidate_sse", "weight"]
        ].sum()
        seeds = grouped.index.get_level_values("seed").unique()
        blocks = grouped.index.get_level_values("block").unique()
        index = pd.MultiIndex.from_product([seeds, blocks], names=["seed", "block"])
        totals = grouped.reindex(index).to_numpy().reshape(len(seeds), len(blocks), 3)
        if not np.isfinite(totals).all() or (totals[..., 2] <= 0).any():
            raise ValueError("Missing seed/block coverage")
        overall = totals.sum(axis=(0, 1))
        if overall[0] <= 0:
            raise ValueError("Cannot compute relative skill against zero control error")
        resampled = []
        for _ in range(draws):
            si = rng.integers(len(seeds), size=len(seeds))
            bi = rng.integers(len(blocks), size=len(blocks))
            sample = totals[si[:, None], bi[None, :]].sum(axis=(0, 1))
            resampled.append(100 * (1 - np.sqrt(sample[1] / sample[0])))
        low, high = np.percentile(resampled, [2.5, 97.5])
        result.append(
            {
                "region": region,
                "nominal_lead_days": int(step) * 5,
                "control_vector_rmse": float(np.sqrt(overall[0] / overall[2])),
                "candidate_vector_rmse": float(np.sqrt(overall[1] / overall[2])),
                "improvement_percent": float(
                    100 * (1 - np.sqrt(overall[1] / overall[0]))
                ),
                "paired_block_ci_low": float(low),
                "paired_block_ci_high": float(high),
                "seed_pairs": len(seeds),
                "quarter_blocks": len(blocks),
                "dates_per_seed": len(group) // len(seeds),
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", nargs="+", required=True)
    parser.add_argument("--candidate", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    results = compare(args.control, args.candidate)
    args.output.write_text(
        json.dumps(
            {
                "comparisons": results,
                "uncertainty": "Paired seeds and calendar-quarter blocks; few test years limit interannual claims.",
            },
            indent=2,
        )
        + "\n"
    )
    for row in results:
        print(
            f"{row['region']:25s} {row['nominal_lead_days']:2d}d: {row['improvement_percent']:+.2f}% RMSE improvement [{row['paired_block_ci_low']:+.2f}, {row['paired_block_ci_high']:+.2f}]"
        )


if __name__ == "__main__":
    main()
