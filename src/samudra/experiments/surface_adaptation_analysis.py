# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Audit and compare completed adaptation arms; never select or train a model."""

import argparse
import csv
import gzip
import io
import itertools
import json
from pathlib import Path

import numpy as np

MODES = (
    "inferred",
    "true",
    "inferred_persistence",
    "climatology",
    "frozen_pretrain",
    "wave1_joint",
)
REGIONS = ("global", "tropics", "extratropics")
LEADS = (5, 10, 15, 20, 25, 30)
VARIABLES = ("thetao", "so", "uo", "vo", "zos", "sst")
ORIGIN_KEY = ("mode", "region", "lead_days", "variable", "origin_index", "origin_time")


def read_bytes(path):
    """Read original bytes from raw, gzip, or bounded-size gzip parts."""
    if path.exists():
        return path.read_bytes()
    compressed = Path(str(path) + ".gz")
    if compressed.exists():
        return gzip.decompress(compressed.read_bytes())
    parts = sorted(path.parent.glob(path.name + ".gz.part[0-9][0-9][0-9]"))
    if not parts:
        raise FileNotFoundError(path)
    expected = [Path(str(path) + f".gz.part{index:03d}") for index in range(len(parts))]
    if parts != expected:
        raise ValueError(f"Missing or out-of-order compressed parts: {path}")
    return gzip.decompress(b"".join(part.read_bytes() for part in parts))


def read_csv(path):
    with io.StringIO(read_bytes(path).decode(), newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def unique_rows(rows, fields):
    result = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in result:
            raise ValueError(f"Duplicate metric key: {key}")
        result[key] = row
    return result


def paired_year_bootstrap(candidate, reference, dates, *, replicates=5000, seed=1729):
    """Paired calendar-year blocks; origin-weighted RMSE in each bootstrap sample."""
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    dates = np.asarray(dates)
    if candidate.shape != reference.shape or candidate.shape != dates.shape:
        raise ValueError("Candidate, reference and dates must be paired one-to-one")
    if candidate.ndim != 1 or not len(candidate):
        raise ValueError("Expected nonempty one-dimensional MSE arrays")
    if not np.isfinite(candidate).all() or not np.isfinite(reference).all():
        raise ValueError("Nonfinite MSE")
    if (candidate < 0).any() or (reference < 0).any() or reference.mean() <= 0:
        raise ValueError(
            "RMSE comparison requires nonnegative MSE and positive reference"
        )
    years = np.array([str(date)[:4] for date in dates])
    blocks = np.unique(years)
    if len(blocks) < 2:
        raise ValueError("At least two calendar-year blocks are required")
    counts = np.array([(years == year).sum() for year in blocks])
    candidate_sum = np.array([candidate[years == year].sum() for year in blocks])
    reference_sum = np.array([reference[years == year].sum() for year in blocks])
    draws = np.random.default_rng(seed).integers(
        0, len(blocks), (replicates, len(blocks))
    )
    n = counts[draws].sum(axis=1)
    candidate_mse = candidate_sum[draws].sum(axis=1) / n
    reference_mse = reference_sum[draws].sum(axis=1) / n
    if (reference_mse <= 0).any():
        raise ValueError("Zero reference error in a resampled year block")
    gains = 100 * (1 - np.sqrt(candidate_mse / reference_mse))
    low, high = np.quantile(gains, [0.025, 0.975])
    return {
        "rmse_reduction_pct": 100 * (1 - np.sqrt(candidate.mean() / reference.mean())),
        "ci95_low_pct": float(low),
        "ci95_high_pct": float(high),
        "year_blocks": len(blocks),
        "origins": len(candidate),
    }


def load_arm(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    marker = json.loads((directory / "COMPLETE.json").read_text())
    initialization = json.loads((directory / "initialization.json").read_text())
    if manifest["arguments"]["max_steps"] or manifest["world_size"] != 4:
        raise ValueError(f"Not a four-GPU production evaluation: {directory}")
    if marker["arm"] != manifest["arguments"]["arm"]:
        raise ValueError(f"Completion marker mismatch: {directory}")
    rows = []
    for rank in range(4):
        rows.extend(read_csv(directory / f"heldout-origins-rank{rank}.csv"))
    unique_rows(rows, ORIGIN_KEY)
    origins = sorted(
        {(r["origin_index"], r["origin_time"]) for r in rows}, key=lambda x: int(x[0])
    )
    if len(origins) != 99:
        raise ValueError(f"Expected 99 origins in {directory}, found {len(origins)}")
    expected = {
        (mode, region, str(lead), variable, index, date)
        for mode, region, lead, variable, (index, date) in itertools.product(
            MODES, REGIONS, LEADS, VARIABLES, origins
        )
    }
    if set(unique_rows(rows, ORIGIN_KEY)) != expected or len(rows) != 64152:
        raise ValueError(f"Incomplete origin grid: {directory}")
    measures = (
        "normalized_mse",
        "physical_mse",
        "prediction_second_moment",
        "target_second_moment",
    )
    for row in rows:
        if row["evaluation"] != "heldout":
            raise ValueError("Non-heldout row in final metrics")
        for measure in measures:
            value = float(row[measure])
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {measure} in {directory}")
    lookup = unique_rows(rows, ORIGIN_KEY)
    groups = {}
    for mode, region, lead, variable in itertools.product(
        MODES, REGIONS, LEADS, VARIABLES
    ):
        selected = [
            lookup[(mode, region, str(lead), variable, index, date)]
            for index, date in origins
        ]
        groups[(mode, region, lead, variable)] = {
            measure: np.array([float(r[measure]) for r in selected])
            for measure in measures
        }
    channels = read_csv(directory / "heldout_metrics.csv")
    channel_lookup = unique_rows(channels, ("mode", "region", "lead_days", "channel"))
    expected_channels = set(
        itertools.product(MODES, REGIONS, map(str, LEADS), manifest["channels"])
    )
    if set(channel_lookup) != expected_channels or len(channels) != 8316:
        raise ValueError(f"Incomplete channel grid: {directory}")
    for row in channels:
        if int(row["origins"]) != 99:
            raise ValueError("Incorrect aggregate origin count")
    for key, group in groups.items():
        mode, region, lead, variable = key
        names = [
            name
            for name in manifest["channels"]
            if ("sst" if name == "thetao_0" else name.split("_")[0]) == variable
        ]
        selected = [channel_lookup[(mode, region, str(lead), name)] for name in names]
        for origin_field, aggregate_field, squared in (
            ("normalized_mse", "normalized_rmse", True),
            ("physical_mse", "physical_rmse", True),
            ("prediction_second_moment", "prediction_second_moment", False),
            ("target_second_moment", "target_second_moment", False),
        ):
            aggregate = np.array([float(r[aggregate_field]) for r in selected])
            if squared:
                aggregate = aggregate**2
            np.testing.assert_allclose(
                group[origin_field].mean(), aggregate.mean(), rtol=2e-5, atol=1e-10
            )
    for mode, region, lead in itertools.product(MODES, REGIONS, LEADS):
        groups[(mode, region, lead, "ts")] = {
            "normalized_mse": (
                groups[(mode, region, lead, "thetao")]["normalized_mse"]
                + groups[(mode, region, lead, "so")]["normalized_mse"]
            )
            / 2
        }
    return manifest, marker, initialization, origins, groups


def analyze(root, output, arms, replicates):
    if not {"A", "B", "C", "D"}.issubset(arms):
        raise ValueError("The primary comparison requires all four approved arms")
    results = {arm: load_arm(root / arm) for arm in arms}
    first = results[arms[0]]
    expected_arms = {
        "A": ("initializer", 1729),
        "B": ("evolution", 1729),
        "C": ("joint", 1729),
        "D": ("joint", 1730),
    }
    training = []
    for arm, (manifest, marker, initialization, origins, groups) in results.items():
        arguments = manifest["arguments"]
        if arm in expected_arms:
            expected_arm, expected_seed = expected_arms[arm]
            if (arguments["arm"], arguments["seed"]) != (expected_arm, expected_seed):
                raise ValueError(f"Incorrect primary arm/seed: {arm}")
            if (
                arguments["learning_rate"] != 1e-5
                or arguments["reconstruction_weight"] != 0.1
            ):
                raise ValueError(f"Incorrect primary adaptation losses/rate: {arm}")
        for field in ("data", "channels", "depths_m", "normalization", "boundary"):
            if manifest[field] != first[0][field]:
                raise ValueError(
                    f"Inconsistent data/evaluation configuration {field}: {arm}"
                )
        if arguments["batch_size"] != 2:
            raise ValueError(f"Incorrect per-rank batch size: {arm}")
        progress = [
            json.loads(line)
            for line in read_bytes(root / arm / "progress.jsonl").decode().splitlines()
        ]
        validation = [
            row for row in progress if row.get("phase") == "adapt" and "ts_mse" in row
        ]
        selected = [
            row
            for row in progress
            if row.get("event") == "selected_checkpoint_validation"
        ][-1]
        np.testing.assert_allclose(
            selected["ts_mse"], marker["selected_ts_mse"], rtol=1e-6
        )
        np.testing.assert_allclose(
            selected["ts_mse"], min(row["ts_mse"] for row in validation), rtol=1e-6
        )
        finished = [row for row in validation if row.get("phase_complete")][-1]
        if arguments["arm"] != "joint" and not finished["frozen_state_verified"]:
            raise ValueError(f"Missing frozen-state verification: {arm}")
        training.append(
            dict(
                arm=arm,
                starting_ts_mse=validation[0]["ts_mse"],
                selected_ts_mse=selected["ts_mse"],
                final_step=finished["step"],
                elapsed_seconds=finished["elapsed_seconds"],
                early_stopped=finished["early_stopped"],
                selected_reconstruction_ts_mse=selected["reconstruction_ts_mse"],
                selected_forecast_full_mse=selected["forecast_full_mse"],
            )
        )
        if origins != first[3]:
            raise ValueError(f"Mismatched paired origins: {arm}")
        for field in (
            "initial_model_fingerprint",
            "initial_initializer_fingerprint",
            "initial_evolution_fingerprint",
            "pretrain",
            "shared_initializer",
            "wave1_joint",
        ):
            if initialization[field] != first[2][field]:
                raise ValueError(f"Mismatched starting checkpoint {field}: {arm}")
        for key, group in groups.items():
            if key[-1] == "ts":
                continue
            np.testing.assert_allclose(
                group["target_second_moment"],
                first[4][key]["target_second_moment"],
                rtol=2e-5,
                atol=1e-10,
            )
            if key[0] in ("frozen_pretrain", "wave1_joint", "climatology"):
                for measure, values in group.items():
                    np.testing.assert_allclose(
                        values, first[4][key][measure], rtol=2e-5, atol=1e-10
                    )
    summaries, comparisons = [], []
    dates = [date for _, date in first[3]]
    for arm, (_, _, _, _, groups) in results.items():
        for (mode, region, lead, variable), group in groups.items():
            row = dict(
                arm=arm,
                mode=mode,
                region=region,
                lead_days=lead,
                variable=variable,
                normalized_rmse=float(np.sqrt(group["normalized_mse"].mean())),
            )
            row["physical_rmse"] = (
                float(np.sqrt(group["physical_mse"].mean())) if variable != "ts" else ""
            )
            row["physical_second_moment_ratio"] = (
                float(
                    group["prediction_second_moment"].mean()
                    / group["target_second_moment"].mean()
                )
                if variable != "ts" and group["target_second_moment"].mean() > 0
                else ""
            )
            summaries.append(row)
        references = [
            (arm, mode)
            for mode in (
                "frozen_pretrain",
                "wave1_joint",
                "inferred_persistence",
                "true",
            )
        ]
        if arm in ("B", "C", "D"):
            references += [
                (other, "inferred")
                for other in arms
                if other < arm and other in ("A", "B", "C")
            ]
        if arm not in ("A", "B", "C", "D"):
            references.append(("C", "inferred"))
        for reference_arm, mode in references:
            reference_groups = results[reference_arm][4]
            for region, lead, variable in itertools.product(
                REGIONS, LEADS, (*VARIABLES, "ts")
            ):
                candidate = groups[("inferred", region, lead, variable)][
                    "normalized_mse"
                ]
                reference = reference_groups[(mode, region, lead, variable)][
                    "normalized_mse"
                ]
                comparisons.append(
                    dict(
                        arm=arm,
                        reference_arm=reference_arm,
                        reference_mode=mode,
                        region=region,
                        lead_days=lead,
                        variable=variable,
                        **paired_year_bootstrap(
                            candidate, reference, dates, replicates=replicates
                        ),
                    )
                )
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "training_summary.csv", training)
    write_csv(output / "grouped_metrics.csv", summaries)
    write_csv(output / "paired_comparisons.csv", comparisons)
    audit = {
        "arms": {
            arm: {
                "manifest": result[0],
                "completion": result[1],
                "initialization": result[2],
            }
            for arm, result in results.items()
        },
        "origins": first[3],
        "channel_rows_per_arm": 8316,
        "origin_rows_per_arm": 64152,
        "bootstrap": {
            "method": "paired calendar-year blocks, origin-weighted RMSE",
            "replicates": replicates,
            "seed": 1729,
        },
        "limitations": "Descriptive intervals over nine partly sampled calendar years; no multiple-comparison adjustment. Same pretrained model, not independent pretraining-seed replication. No observational or daily-output skill claim.",
    }
    (output / "analysis-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return summaries, comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", default=["A", "B", "C", "D"])
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    args = parser.parse_args()
    if args.bootstrap_replicates < 1000:
        parser.error("Use at least 1000 bootstrap replicates")
    analyze(args.input, args.output, args.arms, args.bootstrap_replicates)


if __name__ == "__main__":
    main()
