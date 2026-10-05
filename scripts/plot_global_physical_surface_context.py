#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Add date-matched OM4 SST/SSH context to the three endpoint rollout panels."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from plot_observation_missingness import (  # type: ignore[import-not-found]
    FIELDS,
    panel_plot,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--om4", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    grid = dict(np.load(args.arrays / "grid.npz"))
    meta = json.loads((args.arrays / "COMPLETE.json").read_text())
    refs = json.loads((args.om4 / "COMPLETE.json").read_text())
    assert refs["metadata_sha256"] == meta["om4"]["metadata_sha256"]
    assert refs["fields"] == ["thetao_0", "zos"]
    stages = [
        ("obs08000", "Final: 8,000 obs"),
        ("scratch08000", "Obs-only: 8,000 obs"),
        ("scratch16000", "Obs-only: 16,000 obs"),
    ]
    audit = {
        "reference": refs,
        "maps": {},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    for lead, state_index, ref_index in [(30, 0, 5), (365, 1, 72)]:
        path = args.om4 / f"om4-day{lead}.npz"
        receipt = refs["files"][path.name]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt["sha256"]
        assert path.stat().st_size == receipt["bytes"]
        sample = dict(np.load(path))
        for axis in ["lat", "lon"]:
            np.testing.assert_array_equal(sample[axis], grid[axis])
        np.testing.assert_array_equal(sample["fields"], grid["names"][[0, 6]])
        np.testing.assert_array_equal(
            sample["origins"], ["2015-01-01", "2018-01-01", "2021-01-01"]
        )
        for index, origin_value in enumerate(sample["origins"]):
            origin = str(origin_value)
            arrays = {
                key: dict(np.load(args.arrays / f"{key}-{origin}.npz"))
                for key, _ in stages
            }
            interval = next(
                v
                for v in refs["alignment"]
                if v["origin"] == origin and v["lead_days"] == lead
            )
            start = pd.Timestamp(origin) + pd.Timedelta(days=lead - 5)
            end = start + pd.Timedelta(days=5)
            assert interval["complete"] and interval["covered_days"] == 5
            assert interval["start"] == str(start.date()) and interval["end"] == str(
                end.date()
            )
            truth = np.full_like(arrays["obs08000"]["initial"], np.nan)
            truth[[0, 6]] = arrays["obs08000"]["reference"][ref_index, :2]
            context = np.full_like(truth, np.nan)
            context[[0, 6]] = sample["surface"][index]
            panels = [
                (label, arrays[key]["state"][state_index], grid)
                for key, label in stages
            ]
            panels += [
                ("Observations: matching five-day bin", truth, grid),
                (
                    f"OM4 model-data sample: {start.date()} to {(end - pd.Timedelta(days=1)).date()}",
                    context,
                    grid,
                ),
            ]
            for field in FIELDS:
                if field[0] not in [0, 6]:
                    continue
                filename = f"{origin}-day{lead}-channel{field[0]}.png"
                audit["maps"][filename] = panel_plot(
                    panels,
                    f"{field[1]}: {origin}, day{lead} (gray: model land; white: missing)",
                    field,
                    args.output / filename,
                    distinguish_missing=True,
                )
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
