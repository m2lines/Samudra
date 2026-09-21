#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Audit and summarize collected wave-three per-channel evaluation rows."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

VARIABLES = ("thetao", "so", "uo", "vo", "zos", "sst")
MODES = ("inferred", "true", "inferred_persistence", "true_persistence")
REGIONS = ("global", "tropics", "extratropics")


def summarize(root, expected, output):
    expected = json.loads(Path(expected).read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    depth_records = []
    audit = {}
    references: dict[str, pd.Series] = {}
    for run in sorted(Path(root).iterdir()):
        files = sorted(run.glob("heldout-rank*.csv")) if run.is_dir() else []
        if not files:
            continue
        completion = json.loads((run / "EVAL_COMPLETE.json").read_text())
        manifest = json.loads((run / "manifest.json").read_text())
        names = manifest["channels"]
        assert len(files) == completion["ranks"]
        frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
        keys = ["mode", "region", "origin_time", "lead_days", "channel"]
        expected_rows = len(MODES) * len(REGIONS) * len(expected) * 7 * len(names)
        assert len(frame) == expected_rows, (run, len(frame), expected_rows)
        assert not frame.duplicated(keys).any(), run
        assert set(frame["mode"]) == set(MODES)
        assert set(frame["region"]) == set(REGIONS)
        assert set(frame["origin_time"]) == set(expected)
        assert set(frame["lead_days"]) == set(range(0, 31, 5))
        assert set(frame["channel"]) == set(names)
        assert np.isfinite(frame[["normalized_mse", "physical_mse"]]).all().all()
        assert (frame[["normalized_mse", "physical_mse"]] >= 0).all().all()
        zero = frame[
            (frame["mode"].isin(["true", "true_persistence"]))
            & (frame["lead_days"] == 0)
        ]
        assert zero["normalized_mse"].max() == 0
        for mode in ["true_persistence", "true"]:
            # Joint adaptation changes true evolution, never true persistence.
            if mode == "true" and manifest["arguments"]["phase"] == "joint":
                continue
            ref = (
                frame[frame["mode"] == mode]
                .set_index(keys)["normalized_mse"]
                .sort_index()
            )
            if mode in references:
                np.testing.assert_allclose(ref, references[mode], rtol=1e-5, atol=1e-8)
            else:
                references[mode] = ref
        depths = (
            frame.groupby(["mode", "region", "lead_days", "channel"])[
                ["normalized_mse", "physical_mse"]
            ]
            .mean()
            .reset_index()
        )
        depths["run"] = run.name
        depth_records.append(depths)
        for variable in VARIABLES:
            channels = [
                n
                for n in names
                if (
                    n == "thetao_0"
                    if variable == "sst"
                    else n != "thetao_0"
                    and (n == variable or n.startswith(variable + "_"))
                )
            ]
            selected = frame[frame["channel"].isin(channels)]
            grouped = (
                selected.groupby(["mode", "region", "origin_time", "lead_days"])[
                    ["normalized_mse", "physical_mse"]
                ]
                .mean()
                .reset_index()
            )
            grouped["run"] = run.name
            grouped["variable"] = variable
            records.append(grouped)
        audit[run.name] = {
            "origins": len(expected),
            "rows": len(frame),
            "first_origin": min(expected),
            "last_origin": max(expected),
            "phase": manifest["arguments"]["phase"],
        }
    if not records:
        raise ValueError("No complete evaluation products found")
    origins = pd.concat(records, ignore_index=True)
    ts = (
        origins[origins["variable"].isin(["thetao", "so"])]
        .groupby(["run", "mode", "region", "origin_time", "lead_days"])[
            "normalized_mse"
        ]
        .mean()
        .reset_index()
    )
    ts["variable"] = "ts"
    origins = pd.concat([origins, ts], ignore_index=True)
    origins.to_csv(output / "origins.csv", index=False)
    pd.concat(depth_records, ignore_index=True).to_csv(
        output / "depth-summary.csv", index=False
    )
    means = (
        origins.groupby(["run", "mode", "region", "lead_days", "variable"])[
            ["normalized_mse", "physical_mse"]
        ]
        .mean()
        .reset_index()
    )
    means["normalized_rmse"] = np.sqrt(means["normalized_mse"])
    means["physical_rmse"] = np.sqrt(means["physical_mse"])
    means.to_csv(output / "summary.csv", index=False)
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", required=True)
    p.add_argument(
        "--expected-origins", default="docs/experiments/surface-wave3-origins.json"
    )
    p.add_argument("--output", required=True)
    args = p.parse_args()
    summarize(args.raw, args.expected_origins, args.output)


if __name__ == "__main__":
    main()
