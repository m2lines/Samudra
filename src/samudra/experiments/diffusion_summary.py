# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Aggregate completed A/B observation reports with paired year-block uncertainty."""

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np

from samudra.experiments.observation_metrics import selection_score

RUNS = ("A-1729", "A-1730", "B-1729", "B-1730")
YEARS = tuple(str(year) for year in range(2015, 2023))


def pool(records, group, key):
    """Sum additive spatial scores over origins and all available forecast bins."""
    result = []
    for record in records:
        array = np.asarray(record["statistics"][group][key], dtype=np.float64)
        if key == "rank_weights":
            result.append(array.reshape(array.shape[0], -1, array.shape[-1]).sum(1))
        else:
            result.append(array.reshape(-1, array.shape[-1]).sum(0))
    return np.stack(result).sum(0)


def reduce_records(records, group):
    weights = pool(records, group, "weight")
    keep = weights > 0
    fields = records[0]["statistics"][group]
    values = {}
    profiles = {}
    for key in fields:
        if key in ("weight", "cells", "members"):
            continue
        numerator = pool(records, group, key)
        per_channel = numerator[..., keep] / weights[keep]
        values[key] = per_channel.mean(-1).tolist()
        profiles[key] = per_channel.tolist()
    return dict(
        supported_channels=np.flatnonzero(keep).tolist(),
        scores=values,
        profiles=profiles,
    )


def paired_year_bootstrap(records, *, draws=2000, seed=7163):
    """Resample years jointly across models/seeds; never treat cells as replicates."""
    arrays = {}
    for name in RUNS:
        group, metric = (
            ("point_mass_interior", "absolute_error")
            if name.startswith("A")
            else ("interior", "fair_crps")
        )
        groups = [
            [row for row in records[name] if row["origin"].startswith(year)]
            for year in YEARS
        ]
        if any(len(rows) != 12 for rows in groups):
            raise ValueError("Year-block analysis requires all 12 months in every year")
        arrays[name] = (
            np.stack([pool(rows, group, metric) for rows in groups]),
            np.stack([pool(rows, group, "weight") for rows in groups]),
        )
    for seed_name in ("1729", "1730"):
        np.testing.assert_array_equal(
            arrays["A-" + seed_name][1], arrays["B-" + seed_name][1]
        )

    def estimate(indices):
        scores = {}
        for name, (numerator, denominator) in arrays.items():
            weight = denominator[indices].sum(0)
            keep = weight > 0
            scores[name] = (numerator[indices].sum(0)[keep] / weight[keep]).mean()
        deterministic = np.mean([scores["A-1729"], scores["A-1730"]])
        diffusion = np.mean([scores["B-1729"], scores["B-1730"]])
        return float(1 - diffusion / deterministic)

    rng = np.random.default_rng(seed)
    values = [estimate(rng.integers(0, 8, size=8)) for _ in range(draws)]
    return dict(
        fractional_interior_crps_improvement=estimate(np.arange(8)),
        percentile_95_interval=np.quantile(values, [0.025, 0.975]).tolist(),
        per_seed_fractional_improvement={
            seed_name: float(
                1
                - reduce_records(records["B-" + seed_name], "interior")["scores"][
                    "fair_crps"
                ]
                / reduce_records(records["A-" + seed_name], "point_mass_interior")[
                    "scores"
                ]["absolute_error"]
            )
            for seed_name in ("1729", "1730")
        },
        draws=draws,
        seed=seed,
        scope="Paired calendar-year resampling, conditional on these two fitted seeds and fixed evaluation draws in this reused development cohort; excludes training and Monte Carlo uncertainty",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--upstream-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    upstream = json.loads(gzip.decompress(args.upstream_evidence.read_bytes()))
    reference = upstream["selection-reference"]
    baseline = upstream["transfer-main-monthly-test/selected.json"]
    output: dict[str, Any] = dict(
        scope="Completed A/B first wave; later campaign stages remain pending",
        aggregation="Pool wet-area numerators and denominators over origins and available bins, divide per supported channel, then average channels. CRPS/MSE use fixed observation-standardized units. Annual entries average per-origin RMSE.",
        reference_training="Effective batch8,1k reconstruction+8k joint, trainable dynamics; A/B batch1,6k observation updates,frozen dynamics. Reference is not a matched treatment control.",
        interval_caution="Eight-member empirical 80/90% intervals do not have nominal coverage even for calibrated exchangeable draws. Use rank reference1/9; expected full min/max coverage7/9.",
        baseline=dict(
            checkpoint_prefix=baseline["sha256"][:16],
            composite=selection_score(
                baseline["metrics"], reference["control"], reference["spectral_keys"]
            ),
            point=baseline["metrics"],
            annual={
                origin: upstream[f"transfer-main-annual-test/{origin}.json"]
                for origin in ("2015-01-01", "2018-01-01", "2021-01-01")
            },
        ),
        runs={},
    )
    all_records = {}
    rows = []
    for name in RUNS:
        root = args.reports / name
        protocol = json.loads((root / "protocol.json").read_text())
        completed = json.loads((root / "MONTHLY_REPORT_COMPLETE.json").read_text())
        annual = json.loads((root / "annual/COMPLETE.json").read_text())
        if completed != dict(
            protocol,
            member_structure="complete",
            native_grid_member_exports="complete",
            calibration="complete"
            if name.startswith("B")
            else "not_applicable_deterministic",
        ):
            raise ValueError("Monthly completion differs from report contract")
        if annual["inputs"]["checkpoint_sha256"] != protocol["checkpoint_sha256"]:
            raise ValueError("Annual checkpoint differs from monthly")
        if protocol["origins"] != baseline["metrics"]["origins"]:
            raise ValueError("Reference and A/B reporting origins differ")
        if (
            protocol["training_protocol"]["observation_manifest_sha256"]
            != upstream["transfer-main-monthly-test/evaluation-input.json"][
                "data_manifest_sha256"
            ]
        ):
            raise ValueError("Reference and A/B data differ")
        records = [
            json.loads(path.read_text())
            for path in sorted((root / "calibration").glob("*.json"))
        ]
        if [row["origin"] for row in records] != protocol["origins"]:
            raise ValueError("Incomplete calibration/point-score cohort")
        all_records[name] = records
        point = json.loads((root / "point-metrics.json").read_text())
        groups = {
            group: reduce_records(records, group) for group in records[0]["statistics"]
        }
        annual_records = {
            origin: json.loads((root / "annual" / (origin + ".json")).read_text())
            for origin in ("2015-01-01", "2018-01-01", "2021-01-01")
        }
        output["runs"][name] = dict(
            checkpoint_prefix=protocol["checkpoint_sha256"][:16],
            evaluator=protocol["evaluator_commit"][:12],
            members=protocol["members"],
            composite=point["reporting_score"],
            point=point["metrics"],
            probabilistic=groups,
            annual=annual_records,
        )
        for group, value in groups.items():
            for metric, score in value["scores"].items():
                if isinstance(score, list):
                    continue
                rows.append(dict(run=name, group=group, metric=metric, value=score))
    output["paired_year_bootstrap"] = paired_year_bootstrap(all_records)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    (args.output / "evidence.json.gz").write_bytes(
        gzip.compress(json.dumps(output, sort_keys=True).encode(), mtime=0)
    )
    with (args.output / "probabilistic-scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["run", "group", "metric", "value"], lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(output["paired_year_bootstrap"], indent=2))


if __name__ == "__main__":
    main()
