# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Offline, frozen-cohort common-grid reporting; never runs a forecast.

JSON configuration: mode (annual/monthly), split (test/validation), origins,
native_grid, legacy_data, annual_data (annual only), and models. Each model has
name, format (common/legacy), evaluation directory, checkpoint_sha256, and
artifacts mapping relative filenames to expected SHA256. Monthly models also
have label (selected or fixed-budget). All paths resolve relative to the config.

Pin COMPLETE.json, input.json (annual) or evaluation-input.json (monthly), and
the consumed NPZs in artifacts; monthly reports also require <label>.json.
Common sidecars additionally must match COMPLETE's embedded checksum inventory.
Legacy candidate exports supply predictions only. Canonical truth and training
climatology always come from legacy_data/annual_data, on one fixed geometry.
Monthly reporting holds cross-origin arrays for the established EKE definition;
annual reporting processes one origin at a time. Both are CPU-only.
"""

import argparse
import csv
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from samudra.experiments.observation_annual import read_origin
from samudra.experiments.observation_common_grid import CommonGrid
from samudra.experiments.observation_data import SPLITS
from samudra.experiments.observation_pilot import atomic_json, digest
from samudra.experiments.observation_training import Samples

COMPONENTS = ("sst_rmse", "velocity_rmse", "ohc_0_700_rmse", "ohc_700_2000_rmse")
CONTROL = "Training seasonal climatology"


def relative_path(root, name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Artifact names must be relative without traversal")
    return Path(root) / path


class Audit:
    """Hash every consumed artifact, checking immutable upstream expectations."""

    def __init__(self):
        self.files = {}

    def file(self, path, expected=None):
        path = Path(path).resolve()
        actual = digest(path)
        if expected is not None and actual != expected:
            raise ValueError(f"Artifact checksum differs: {path}")
        if str(path) in self.files and self.files[str(path)] != actual:
            raise ValueError(f"Artifact changed during collection: {path}")
        self.files[str(path)] = actual
        return path

    def json(self, path, expected=None):
        return json.loads(self.file(path, expected).read_text())


class Candidate:
    def __init__(self, model, mode, split, origins, geometry, audit):
        self.model, self.mode, self.geometry, self.audit = model, mode, geometry, audit
        if model["format"] not in ("common", "legacy"):
            raise ValueError("Candidate format must be common or legacy")
        self.root = Path(model["evaluation"])
        self.origins = origins
        self.complete = self.json("COMPLETE.json")
        self.signature = self.json(
            "input.json" if mode == "annual" else "evaluation-input.json"
        )
        if (
            self.signature["checkpoint_sha256"] != model["checkpoint_sha256"]
            or self.signature["split"] != split
            or self.signature.get("global_observations") is not True
        ):
            raise ValueError("Candidate checkpoint, split or global domain differs")
        if mode == "annual":
            if (
                self.complete["inputs"] != self.signature
                or sorted(self.complete["origins"]) != origins
            ):
                raise ValueError("Annual candidate cohort/signature differs")
        else:
            if (
                self.complete["origins"] != len(origins)
                or self.complete["split"] != split
            ):
                raise ValueError("Monthly candidate cohort differs")
            if model.get("label") not in ("selected", "fixed-budget"):
                raise ValueError("Monthly label must be selected or fixed-budget")
            # This receipt establishes checkpoint identity even for legacy NPZs.
            metrics = self.json(model["label"] + ".json")
            if (
                metrics["sha256"] != model["checkpoint_sha256"]
                or metrics["metrics"]["origins"] != origins
            ):
                raise ValueError("Monthly candidate checkpoint/origin order differs")

    def path(self, name):
        if name not in self.model["artifacts"]:
            raise ValueError(f"Missing pinned artifact checksum: {name}")
        return self.audit.file(
            relative_path(self.root, name), self.model["artifacts"][name]
        )

    def json(self, name):
        return json.loads(self.path(name).read_text())

    def sidecar(self, origin, months):
        name = (
            f"{origin}.npz"
            if self.mode == "annual"
            else f"{self.model['label']}/{origin}.npz"
        )
        path = self.path("common-grid/" + name)
        if digest(path) != self.complete["common_grid_sidecars"].get(name):
            raise ValueError("Common sidecar differs from completion receipt")
        with np.load(path) as saved:
            for key, value in self.geometry.geometry_arrays().items():
                np.testing.assert_array_equal(saved[key], value, err_msg=key)
            np.testing.assert_array_equal(saved["origins"], [origin])
            np.testing.assert_array_equal(saved["months"], months)
            return saved["prediction"], saved["predicted_ohc"]

    def verify_legacy_truth(self, saved, reference, reference_ohc):
        # Verify the archive's data identity, but score canonical truth below.
        # Old OHC exports may mask dry columns; only our fixed support matters.
        for key, expected, support in (
            ("reference", reference, self.geometry.surface_support),
            ("reference_ohc", reference_ohc, self.geometry.ohc_support),
        ):
            if saved[key].shape != expected.shape:
                raise ValueError("Legacy reference shape differs from canonical truth")
            np.testing.assert_array_equal(
                np.where(support, saved[key], np.nan),
                np.where(support, expected, np.nan),
                err_msg="Legacy archive does not match canonical truth: " + key,
            )

    def annual(self, origin, months, *, reference, reference_ohc):
        if self.model["format"] == "common":
            return self.sidecar(origin, months)
        with np.load(self.path(self.complete["origins"][origin]["arrays"])) as saved:
            np.testing.assert_array_equal(saved["months"], months)
            self.verify_legacy_truth(saved, reference, reference_ohc)
            return saved["surface"], saved["predicted_ohc"]

    def monthly(self, *, reference, reference_ohc):
        if self.model["format"] == "common":
            pairs = [self.sidecar(origin, []) for origin in self.origins]
            return tuple(np.stack([pair[i] for pair in pairs]) for i in (0, 1))
        name = self.model["label"] + "-predictions.npz"
        with np.load(self.path(name)) as saved:
            np.testing.assert_array_equal(saved["origins"], self.origins)
            self.verify_legacy_truth(saved, reference, reference_ohc)
            return saved["prediction"], saved["predicted_ohc"]


def pool_components(records):
    """Equal-origin MSE pooling before square root, including diagnostic ADT."""
    if not records or any(set(row) != set(records[0]) for row in records):
        raise ValueError("Incomplete component cohort")
    if any(not np.isfinite(v) or v < 0 for row in records for v in row.values()):
        raise ValueError("Nonfinite or negative component RMSE")
    return {
        key: float(np.sqrt(np.mean([row[key] ** 2 for row in records])))
        for key in records[0]
    }


def endpoint(record, origin, day):
    if day not in (30, 365) or origin[5:] != "01-01":
        raise ValueError("Endpoint blend requires January starts and day 30/365")
    month = origin[:4] + ("-01" if day == 30 else "-12")
    return {
        **record["leads"][str(day)],
        **{
            "ohc_" + layer + "_rmse": record["monthly_ohc"][month][layer]
            for layer in ("0_700", "700_2000")
        },
    }


def annual_rows(records, origins):
    rows = []
    for day in (30, 365):
        control = pool_components(
            [endpoint(records[CONTROL][o], o, day) for o in origins]
        )
        if any(control[key] <= 0 for key in COMPONENTS):
            raise ValueError("Invalid pooled climatology denominator")
        for name, cases in records.items():
            if sorted(cases) != origins:
                raise ValueError("Incomplete candidate cohort")
            for origin in [*origins, "pooled"]:
                cohort = origins if origin == "pooled" else [origin]
                values = pool_components([endpoint(cases[o], o, day) for o in cohort])
                rows.append(
                    dict(
                        model=name,
                        origin=origin,
                        lead_days=day,
                        rmse_score=float(
                            np.mean([values[k] / control[k] for k in COMPONENTS])
                        ),
                        **values,
                    )
                )
    return rows


def verify_legacy_data(root, split, audit):
    """Require all training months used by climatology and the reporting cohort."""
    checksums = {}
    for line in audit.file(root / "SHA256SUMS").read_text().splitlines():
        checksum, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        relative_path(root, name)
        if name in checksums:
            raise ValueError("Duplicate canonical data checksum")
        checksums[name] = checksum
    required = ["grid.npz", "statistics.npz"]
    for phase in ("train", split):
        months = [str(v) for v in pd.period_range(*SPLITS[phase], freq="M")]
        if sorted(p.stem for p in (root / phase).glob("*.npz")) != months:
            raise ValueError("Incomplete canonical monthly cohort")
        required.extend(f"{phase}/{month}.npz" for month in months)
    for name in required:
        if name not in checksums:
            raise ValueError("Missing canonical data checksum: " + name)
        audit.file(root / name, checksums[name])


def annual_climatology_ohc(data):
    """Exact legacy annual control construction, in normalized model space."""
    values = []
    for month in range(12):
        state = torch.zeros((1, 77, *data.grid["mask"].shape[-2:]))
        indices = data.ts_indices
        state[:, indices] = (
            data.tensor(data.interior_climatology[month]) - data.mean[0, 0, indices]
        ) / data.std[0, 0, indices]
        state[:, [38, 76]] = (
            data.tensor(data.stats["surface_climatology"][month])
            - data.mean[0, 0, [38, 76]]
        ) / data.std[0, 0, [38, 76]]
        values.append(data.ohc(state * data.mask)[0])
    return np.stack(values)


def annual_truth(root, origin, audit):
    directory = root / origin
    manifest = audit.json(directory / "COMPLETE.json")
    if manifest["origin"] != origin:
        raise ValueError("Annual truth origin differs")
    for record in manifest["records"]:
        audit.file(relative_path(directory, record["file"]), record["sha256"])
    _, raw = read_origin(directory)  # Also verifies exact bin times and layout.
    months, ohc = [], []
    for record in manifest["monthly_interiors"]:
        path = audit.file(relative_path(directory, record["file"]), record["sha256"])
        months.append(path.name[:7])
        with np.load(path) as saved:
            ohc.append(saved["ohc"])
    expected = [f"{origin[:4]}-{i:02d}" for i in range(1, 13)]
    if months != expected:
        raise ValueError("Annual canonical calendar-month cohort differs")
    truth = np.concatenate([raw["surface"][19:], raw["velocity"][19:]], axis=1)
    midpoint_months = pd.DatetimeIndex(raw["midpoint"][19:]).month.to_numpy() - 1
    return truth, np.stack(ohc), months, midpoint_months


def collect(config_path, output):
    config_path, output = Path(config_path).resolve(), Path(output)
    if output.exists():
        raise ValueError("Use a fresh output directory; refusing to overwrite a report")
    audit = Audit()
    config = audit.json(config_path)
    mode, split, origins = config["mode"], config["split"], config["origins"]
    if mode not in ("annual", "monthly") or split not in ("test", "validation"):
        raise ValueError("Expected annual/monthly and test/validation")
    if not origins or origins != sorted(set(origins)):
        raise ValueError("Origins must be nonempty, unique and ordered")
    if mode == "monthly" and origins != [
        str(v) for v in pd.period_range(*SPLITS[split], freq="M")
    ]:
        raise ValueError("Monthly report requires the complete canonical split")
    if mode == "annual" and (
        any(o[5:] != "01-01" for o in origins)
        or (split == "test" and origins != ["2015-01-01", "2018-01-01", "2021-01-01"])
        or (split == "validation" and len(origins) != 1)
    ):
        raise ValueError("Annual report requires the established January cohort")

    def resolve(value):
        return (config_path.parent / value).resolve()

    root = resolve(config["legacy_data"])
    verify_legacy_data(root, split, audit)
    data = Samples(root, "cpu", "zero", global_observations=True, cache_size=0)
    data.use_observation_normalization()
    with np.load(audit.file(resolve(config["native_grid"]))) as saved:
        native = dict(saved)
    if data.grid["mask"].shape != (77, 180, 360) or native["mask"].shape != (
        77,
        720,
        1440,
    ):
        raise ValueError("Require canonical native-quarter and legacy-one-degree grids")
    geometry = CommonGrid(native, data.grid)
    names = [model["name"] for model in config["models"]]
    if not names or len(set(names)) != len(names) or CONTROL in names:
        raise ValueError("Missing, duplicate or reserved candidate name")
    candidates = [
        Candidate(
            {**model, "evaluation": str(resolve(model["evaluation"]))},
            mode,
            split,
            origins,
            geometry,
            audit,
        )
        for model in config["models"]
    ]
    for candidate in candidates:
        if candidate.model["format"] == "common" and candidate.signature.get(
            "common_grid_sha256"
        ) != digest(root / "grid.npz"):
            raise ValueError("Common-grid export used a different legacy grid")
    records: dict[str, Any] = {}
    if mode == "annual":
        annual = resolve(config["annual_data"]) / split
        if sorted(p.parent.name for p in annual.glob("*/COMPLETE.json")) != origins:
            raise ValueError("Annual truth cohort differs")
        climate_ohc = annual_climatology_ohc(data)
        records = {name: {} for name in [*names, CONTROL]}
        for origin in origins:
            truth, ohc_truth, months, midpoint_months = annual_truth(
                annual, origin, audit
            )
            common = dict(reference=truth, reference_ohc=ohc_truth, months=months)
            records[CONTROL][origin] = geometry.score_annual(
                data.stats["surface_climatology"][midpoint_months],
                climate_ohc,
                **common,
            )
            for candidate in candidates:
                prediction, ohc = candidate.annual(
                    origin, months, reference=truth, reference_ohc=ohc_truth
                )
                records[candidate.model["name"]][origin] = geometry.score_annual(
                    prediction, ohc, **common
                )
        rows = annual_rows(records, origins)
    else:
        truth, ohc_truth, climates, climate_ohc = [], [], [], []
        for origin in origins:
            sample = data.load(root / split / (origin + ".npz"))
            fields = data.climatology_fields(sample)
            raw = sample["raw"]
            truth.append(
                np.concatenate([raw["surface"][19:25], raw["velocity"][19:25]], 1)
            )
            ohc_truth.append(raw["ohc"])
            climates.append(fields["surface"][0, :6].numpy())
            climate_ohc.append(fields["ohc"][0])
        common = dict(
            reference=np.stack(truth),
            reference_ohc=np.stack(ohc_truth),
            climatology_surface=np.stack(climates),
            climatology_ohc=np.stack(climate_ohc),
        )
        del truth, ohc_truth, climates, climate_ohc, sample, raw, fields
        rows = []
        for candidate in candidates:
            prediction, ohc = candidate.monthly(
                reference=common["reference"], reference_ohc=common["reference_ohc"]
            )
            result = geometry.score(prediction, ohc, **common)
            name = candidate.model["name"]
            records[name] = result
            rows.append(
                dict(
                    model=name,
                    rmse_score=result["common_grid_rmse_ratio"],
                    **{
                        k: result["candidate"]["metrics"][k]
                        for k in (*COMPONENTS, "eke_rmse")
                    },
                )
            )
    # No result is marked complete until every candidate, source and cohort passed.
    output.mkdir(parents=True)
    np.savez_compressed(output / "geometry.npz", **geometry.geometry_arrays())
    atomic_json(
        dict(configuration=config, mode=mode, records=records), output / "results.json"
    )
    with (output / "components.csv").open("w") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(
        dict(
            source_hashes=audit.files,
            code_hashes={
                name: digest(Path(__file__).parent / name)
                for name in (
                    "observation_common_collect.py",
                    "observation_common_grid.py",
                    "observation_training.py",
                    "observation_metrics.py",
                    "observation_annual.py",
                    "observation_prepare.py",
                    "observation_data.py",
                    "../metrics/kernels.py",
                    "../constants.py",
                )
            },
            software_versions={
                name: version(name)
                for name in ("numpy", "scipy", "pandas", "xarray", "torch")
            },
            aggregation=(
                "Equal-origin MSE then sqrt; four-component ratio to pooled legacy training climatology per lead; OHC January/December"
                if mode == "annual"
                else "Established five-component monthly cohort RMSE blend; fixed-lead cross-origin EKE; lead RMSEs averaged over days 5/15/30"
            ),
            support="One native-quarter/legacy intersection, canonical truth validity; no candidate-dependent truth masks",
            selection="Reporting only; no checkpoint selection or native-score changes",
            outputs={
                name: digest(output / name)
                for name in ("geometry.npz", "results.json", "components.csv")
            },
        ),
        output / "COMPLETE.json",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    collect(args.config, args.output)


if __name__ == "__main__":
    main()
