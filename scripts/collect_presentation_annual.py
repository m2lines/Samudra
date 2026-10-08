#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Collect matched fixed-checkpoint annual forecasts and RMSE-only controls."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from samudra.experiments.annual_artifacts import annual_output_hashes
from samudra.experiments.observation_annual import read_origin, surface_metrics
from samudra.experiments.observation_metrics import weighted_rmse
from samudra.experiments.observation_pilot import atomic_json, digest
from samudra.experiments.observation_training import Samples

KEYS = ("sst_rmse", "velocity_rmse", "ohc_0_700_rmse", "ohc_700_2000_rmse")


def endpoint(record, origin, day):
    if day not in (30, 365) or origin[5:] != "01-01":
        raise ValueError("This comparison requires January starts and day 30 or 365")
    month = origin[:4] + ("-01" if day == 30 else "-12")
    return {
        **record["leads"][str(day)],
        **{
            "ohc_" + layer + "_rmse": record["monthly_ohc"][month][layer]
            for layer in ("0_700", "700_2000")
        },
    }


def pooled(records):
    """Equal-origin MSE pooling, followed by square root; not mean RMSE."""
    if not records or any(set(r) != set(records[0]) for r in records):
        raise ValueError("Incomplete metric components")
    if any(not np.isfinite(v) or v < 0 for r in records for v in r.values()):
        raise ValueError("Nonfinite or negative RMSE")
    return {
        k: float(np.sqrt(np.mean([r[k] ** 2 for r in records]))) for k in records[0]
    }


def collect(root, output):
    config = json.loads((root / "paths.json").read_text())
    other_root = Path(config["reuse_annual_root"])
    other_config = json.loads((other_root / "paths.json").read_text())
    if config["origins"] != other_config["origins"]:
        raise ValueError("Annual cohorts differ")
    for key in ("image", "observations", "annual_data"):
        if config[key] != other_config[key]:
            raise ValueError("Annual evaluation configuration differs: " + key)
    locations = [(root, r) for r in config["models"]] + [
        (other_root, r)
        for r in other_config["models"]
        if r["name"] in config["reuse_models"]
    ]
    expected = [r["name"] for r in config["models"]] + config["reuse_models"]
    actual = [r["name"] for _, r in locations]
    if len(set(actual)) != len(actual) or sorted(actual) != sorted(expected):
        raise ValueError("Missing or duplicate presentation model")
    records, metadata = {}, {}
    references: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for location, model in locations:
        folder = location / model["name"]
        verified = json.loads((folder / "VERIFIED.json").read_text())
        if (
            verified["config_sha256"] != digest(location / "paths.json")
            or verified["model"] != model
        ):
            raise ValueError("Model audit contract differs")
        output_hashes = verified["outputs"]
        repaired = None
        if not output_hashes:
            repaired = json.loads((folder / "VERIFIED-OUTPUT-HASHES.json").read_text())
            if repaired["original_verified_sha256"] != digest(folder / "VERIFIED.json"):
                raise ValueError("Output audit repair differs from original record")
            output_hashes = repaired["outputs"]
        complete = json.loads((folder / "COMPLETE.json").read_text())
        if annual_output_hashes(folder, complete) != output_hashes:
            raise ValueError("Missing or changed annual output hashes")
        for filename, checksum in output_hashes.items():
            if digest(folder / filename) != checksum:
                raise ValueError("Annual output hash differs")
        if (
            sorted(complete["origins"]) != config["origins"]
            or complete["inputs"]["checkpoint_sha256"] != model["checkpoint_sha256"]
        ):
            raise ValueError("Annual checkpoint or cohort differs")
        metadata[model["name"]] = dict(
            model=model,
            inputs=complete["inputs"],
            verified=verified,
            repaired_output_audit=repaired,
        )
        for kind in ("metrics", "persistence"):
            label = model["name"] + (
                " / initialized persistence" if kind == "persistence" else ""
            )
            records[label] = {
                origin: json.loads((folder / item[kind]).read_text())
                for origin, item in complete["origins"].items()
            }
        for origin, item in complete["origins"].items():
            with np.load(folder / item["arrays"]) as arrays:
                expected_months = [f"{origin[:4]}-{m:02d}" for m in range(1, 13)]
                np.testing.assert_array_equal(arrays["months"], expected_months)
                if arrays["surface"].shape[0] != 73:
                    raise ValueError("Incomplete continuous annual forecast")
                reference = (arrays["reference"], arrays["reference_ohc"])
                if origin in references:
                    for a, b in zip(reference, references[origin], strict=True):
                        np.testing.assert_array_equal(a, b)
                else:
                    references[origin] = reference

    # Training-only seasonal fields: no test fitting and no forecast reselection.
    data = Samples(config["observations"], "cpu", "zero", global_observations=True)
    data.use_observation_normalization()
    climatology = {}
    ohc_climatology = {}
    for month in (0, 11):
        state = torch.zeros((1, 77, *data.grid["mask"].shape[-2:]))
        indices = data.ts_indices
        state[:, indices] = (
            data.tensor(data.interior_climatology[month]) - data.mean[0, 0, indices]
        ) / data.std[0, 0, indices]
        state[:, [38, 76]] = (
            data.tensor(data.stats["surface_climatology"][month])
            - data.mean[0, 0, [38, 76]]
        ) / data.std[0, 0, [38, 76]]
        ohc_climatology[month] = data.ohc(state * data.mask)[0]
    for origin in config["origins"]:
        _, raw = read_origin(Path(config["annual_data"]) / "test" / origin)
        months = pd.DatetimeIndex(raw["midpoint"][19:]).month.to_numpy() - 1
        prediction = data.stats["surface_climatology"][months]
        truth, ohc_truth = references[origin]
        np.testing.assert_array_equal(
            truth,
            np.concatenate([raw["surface"][19:], raw["velocity"][19:]], axis=1),
        )
        result = surface_metrics(prediction, truth, data)
        result["monthly_ohc"] = {}
        for month in (0, 11):
            values = ohc_climatology[month]
            ohc_reference = np.where(np.isfinite(values), ohc_truth[month], np.nan)
            result["monthly_ohc"][f"{origin[:4]}-{month + 1:02d}"] = {
                layer: weighted_rmse(values[i], ohc_reference[i], data.area.numpy())
                for i, layer in enumerate(("0_700", "700_2000"))
            }
        climatology[origin] = result
    records["Training seasonal climatology"] = climatology
    rows = []
    for day in (30, 365):
        control = pooled([endpoint(climatology[o], o, day) for o in config["origins"]])
        if any(not np.isfinite(control[k]) or control[k] <= 0 for k in KEYS):
            raise ValueError("Invalid RMSE normalization control")
        for model, origins in records.items():
            for origin in [*config["origins"], "pooled"]:
                selected = config["origins"] if origin == "pooled" else [origin]
                values = pooled([endpoint(origins[o], o, day) for o in selected])
                # One shared pooled control per lead, including per-origin rows.
                score = float(np.mean([values[k] / control[k] for k in KEYS]))
                rows.append(
                    dict(
                        model=model,
                        origin=origin,
                        lead_days=day,
                        rmse_score=score,
                        **values,
                    )
                )
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        dict(
            configuration=config,
            collector_sha256=digest(__file__),
            models=metadata,
            metrics=records,
            control_definition="Training monthly mean IAP interior and surface fields; OHC uses calendar-month climatology; surface uses each five-day bin midpoint month",
        ),
        output / "results.json",
    )
    with (output / "rmse-scores.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            dict(
                event="annual_rmse_collected",
                models=len(locations),
                origins=config["origins"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collect(args.root, args.output)
