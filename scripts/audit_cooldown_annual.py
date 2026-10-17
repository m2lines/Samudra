#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit the fixed cooldown annual validation cohort before GPU evaluation."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from samudra.experiments.observation_annual import read_origin, verify_sample_root
from samudra.experiments.observation_checkpoint import load_screen_endpoint
from samudra.experiments.observation_pilot import atomic_json, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text())
    root = Path(plan["annual"])
    origin = root / "validation" / "2013-11-01"
    description, raw = read_origin(origin)
    if (
        description["origin"] != "2013-11-01"
        or not np.isfinite(raw["atmosphere"]).all()
    ):
        raise ValueError("Origin or forcing differs")
    monthly = Path(plan["data"])
    with np.load(monthly / "validation" / "2013-11.npz") as sample:
        for key in ("surface", "velocity", "atmosphere"):
            np.testing.assert_array_equal(raw[key][:25], sample[key][:25])
    months = [str(m) for m in pd.period_range("2013-11", periods=12, freq="M")]
    if [p["file"] for p in description["monthly_interiors"]] != [
        f"{m}-interior.npz" for m in months
    ]:
        raise ValueError("Monthly target inventory differs")
    for entry in description["monthly_interiors"]:
        path = origin / entry["file"]
        if digest(path) != entry["sha256"]:
            raise ValueError("Monthly target hash differs")
        with np.load(path) as values:
            if (
                not np.isfinite(values["ohc"]).any()
                or not np.isfinite(values["values"]).any()
            ):
                raise ValueError("Missing monthly interior targets")
    checkpoints = {}
    for run in plan["runs"]:
        path = Path(run)
        manifest = json.loads((path / "manifest.json").read_text())
        verify_sample_root(monthly, manifest)
        if max(manifest["training_months"]) != "2013-07":
            raise ValueError("Observation training period differs")
        for weights in ("raw", "ema"):
            state, lineage = load_screen_endpoint(path, weights)
            checkpoints[f"{run}/{weights}"] = lineage
            del state
    # Pinned source also supplies the actual recent/early training-window rules.
    # History can overlap OM4 training; forecast targets start after its last date.
    from types import SimpleNamespace

    from samudra.experiments.surface_wave import data_config

    config = data_config(SimpleNamespace(data_root=plan["om4"], readers=4))
    source = config.model_dump(mode="json")["sources"][0]
    if pd.Timestamp(source["train_time"]["end"]) >= pd.Timestamp("2013-11-01"):
        raise ValueError("OM4 training overlaps annual forecast targets")
    atomic_json(
        dict(
            plan_sha256=digest(plan_path),
            origin="2013-11-01",
            annual_manifest_sha256=digest(origin / "COMPLETE.json"),
            exact_monthly_prefix_bins=25,
            forcing_finite=True,
            recent_om4_train_time=source["train_time"],
            early_om4_rule="1958 <= first year; final target year < 1975",
            checkpoints=checkpoints,
        ),
        plan_path.parent / "DATA_AUDITED.json",
    )


if __name__ == "__main__":
    main()
