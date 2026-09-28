# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare completed scratch diffusion reports against the unchanged baseline."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from samudra.experiments.diffusion_summary import pool, reduce_records

ORIGINS = [f"{year}-{month:02}" for year in range(2015, 2023) for month in range(1, 13)]
ANNUAL = ("2015-01-01", "2018-01-01", "2021-01-01")


def summarize_structure(records):
    """Average within-field moments, not variance across dates or ensemble members."""
    result: dict[str, dict[str, list[float | None]]] = {}
    for group in records[0]["statistics"]:
        result[group] = {}
        for metric, support in (
            ("mean", "area"),
            ("variance", "area"),
            ("zonal_increment_mse", "zonal_pair_area"),
            ("meridional_increment_mse", "meridional_pair_area"),
        ):
            numerator, denominator = [], []
            for record in records:
                fields = record["statistics"][group]
                values = np.asarray(fields[metric], dtype=float)
                weights = np.broadcast_to(
                    np.asarray(fields[support], dtype=float), values.shape
                )
                if not np.isfinite(weights).all() or (weights < 0).any():
                    raise ValueError("Invalid structural support")
                valid = weights > 0
                if not np.isfinite(values[valid]).all():
                    raise ValueError("Nonfinite structure on supported cells")
                numerator.append(
                    (np.where(valid, values, 0) * weights)
                    .reshape(-1, values.shape[-1])
                    .sum(0)
                )
                denominator.append(weights.reshape(-1, values.shape[-1]).sum(0))
            total_weight = np.stack(denominator).sum(0)
            total = np.stack(numerator).sum(0)
            result[group][metric] = [
                float(v / w) if w > 0 else None
                for v, w in zip(total, total_weight, strict=True)
            ]
    return result


def read_report(root, *, scratch, architecture="physical"):
    if architecture not in ("physical", "latent"):
        raise ValueError("Unknown forecast architecture")
    protocol = json.loads((root / "protocol.json").read_text())
    complete = json.loads((root / "MONTHLY_REPORT_COMPLETE.json").read_text())
    annual = json.loads((root / "annual/COMPLETE.json").read_text())["inputs"]
    if protocol["origins"] != ORIGINS:
        raise ValueError("Monthly cohort differs")
    if any(complete.get(k) != v for k, v in protocol.items()):
        raise ValueError("Monthly completion contract differs")
    if annual["checkpoint_sha256"] != protocol["checkpoint_sha256"]:
        raise ValueError("Annual checkpoint differs")
    if annual["annual_origins"] != list(ANNUAL):
        raise ValueError("Annual cohort differs")
    if scratch:
        training = protocol["training_protocol"]
        if (
            training.get("initialization") != "scratch"
            or training.get("decoder_width") != 192
        ):
            raise ValueError("Require the authorized from-scratch width-192 model")
        if training["phase"] != "observation":
            raise ValueError("Require observation-fine-tuned diffusion")
        if architecture == "physical":
            if training["arm"] != "B" or complete.get("calibration") != "complete":
                raise ValueError("Incomplete physical-state diffusion report")
        elif (
            training.get("architecture") != "persistent-latent-diffusion"
            or protocol.get("scope") != "observation-report"
        ):
            raise ValueError("Require completed persistent-latent observation report")
        if protocol["members"] != 8:
            raise ValueError("Incomplete eight-member report")
        data_hash = training["observation_manifest_sha256"]
    else:
        if not complete.get("complete") or protocol["members"] != 1:
            raise ValueError("Require completed unchanged deterministic baseline")
        data_hash = protocol["observation_manifest_sha256"]
    records = [
        json.loads(p.read_text()) for p in sorted((root / "calibration").glob("*.json"))
    ]
    if [r["origin"] for r in records] != ORIGINS:
        raise ValueError("Incomplete per-origin score records")
    structures = [
        json.loads(p.read_text()) for p in sorted((root / "structure").glob("*.json"))
    ]
    if [r["origin"] for r in structures] != ORIGINS:
        raise ValueError("Incomplete structural diagnostic cohort")
    point = json.loads((root / "point-metrics.json").read_text())
    summary = dict(
        checkpoint_prefix=protocol["checkpoint_sha256"][:16],
        evaluator=protocol["evaluator_commit"][:12],
        members=protocol["members"],
        composite=point["reporting_score"],
        structure=summarize_structure(structures),
        point=point["metrics"],
        probabilistic={
            group: reduce_records(records, group) for group in records[0]["statistics"]
        },
        annual={
            origin: json.loads((root / "annual" / (origin + ".json")).read_text())
            for origin in ANNUAL
        },
    )
    contract = dict(
        observation_manifest=data_hash, annual_manifests=annual["annual_manifests"]
    )
    return summary, records, contract


def compare_crps(baseline, seeds, *, draws=2000):
    """Paired year-block uncertainty, conditional on fitted seeds and fixed draws."""
    all_records = {"baseline": baseline, **seeds}
    arrays = {}
    for name, records in all_records.items():
        group, score = (
            ("point_mass_interior", "absolute_error")
            if name == "baseline"
            else ("interior", "fair_crps")
        )
        numerators, weights = [], []
        for year in range(2015, 2023):
            rows = [r for r in records if r["origin"].startswith(str(year))]
            if [r["origin"] for r in rows] != [f"{year}-{m:02}" for m in range(1, 13)]:
                raise ValueError("Require each complete ordered year")
            numerators.append(pool(rows, group, score))
            weights.append(pool(rows, group, "weight"))
        arrays[name] = (np.stack(numerators), np.stack(weights))
    for name in seeds:
        np.testing.assert_array_equal(arrays[name][1], arrays["baseline"][1])

    def estimate(indices):
        values = {}
        for name, (numerator, weight) in arrays.items():
            denominator = weight[indices].sum(0)
            keep = denominator > 0
            values[name] = float(
                (numerator[indices].sum(0)[keep] / denominator[keep]).mean()
            )
        reference = values.pop("baseline")
        if reference <= 0:
            raise ValueError("Relative improvement requires positive baseline error")
        return {name: 1 - value / reference for name, value in values.items()}

    paired = estimate(np.arange(8))
    rng = np.random.default_rng(7163)
    bootstrap = [
        np.mean(list(estimate(rng.integers(0, 8, 8)).values())) for _ in range(draws)
    ]
    return dict(
        per_seed_fractional_improvement=paired,
        seed_mean_fractional_improvement=float(np.mean(list(paired.values()))),
        paired_year_percentile_95_interval=np.quantile(
            bootstrap, [0.025, 0.975]
        ).tolist(),
        draws=draws,
        bootstrap_seed=7163,
        scope="Conditional on two fitted seeds and fixed evaluation draws; excludes training and Monte Carlo uncertainty; reused development cohort",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline, base_records, base_contract = read_report(args.baseline, scratch=False)
    runs = {}
    records = {}
    for seed in (1729, 1730):
        name = f"B-{seed}"
        runs[name], records[name], contract = read_report(
            args.reports / name, scratch=True
        )
        if contract != base_contract:
            raise ValueError("Diffusion and baseline use different observation inputs")
    result = dict(
        scope="From-scratch diffusion vs unchanged selected deterministic model",
        reference_training="Baseline has separately trained dynamics; diffusion uses frozen pre-observation dynamics. Not an isolated decoder-only treatment.",
        aggregation="Wet-area pooled over origins/bins, normalized per supported channel, then equal channel average",
        calibration_caution="Eight members: uniform rank reference1/9; full min/max exchangeable coverage7/9; raw quantile intervals lack nominal finite-ensemble coverage",
        baseline=baseline,
        runs=runs,
        interior_crps_comparison=compare_crps(base_records, records),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, sort_keys=True).encode()
    (args.output / "summary.json.gz").write_bytes(gzip.compress(payload, mtime=0))
    print(json.dumps(result["interior_crps_comparison"], indent=2))


if __name__ == "__main__":
    main()
