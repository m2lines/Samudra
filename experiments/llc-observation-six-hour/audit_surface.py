# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Audit matching-date LLC snapshots against daily analyses without bias correction."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--prepared", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
proof = json.loads((args.prepared / "grid.json").read_text())
ds = xr.open_zarr(proof["llc"], chunks=None)
grid = dict(np.load(args.prepared / "grid.npz", allow_pickle=False))
days = pd.date_range("2011-09-14", "2012-09-13", periods=32).normalize()
records = []
for variable, product, channel, unit in (
    ("Theta", "sst", 102, "degC"),
    ("Eta", "adt", 204, "m"),
):
    rows = []
    totals = np.zeros(5, dtype=np.float64)
    for day in days:
        field = ds[variable].sel(time=day)
        if "k" in field.dims:
            field = field.isel(k=0)
        model = field.values.astype(np.float64)
        path = (
            Path(proof["daily_root"]) / product / str(day.year) / f"{day:%Y-%m-%d}.npz"
        )
        with np.load(path, allow_pickle=False) as saved:
            observation = saved["values"][0].astype(np.float64)
        valid = np.isfinite(model) & np.isfinite(observation)
        weights = np.where(valid, grid["wet_area"][channel], 0)
        assert weights.sum() > 0
        model, observation = np.where(valid, model, 0), np.where(valid, observation, 0)
        diff = model - observation
        sums = np.array(
            [
                weights.sum(),
                (weights * model).sum(),
                (weights * observation).sum(),
                (weights * diff).sum(),
                (weights * diff**2).sum(),
            ]
        )
        totals += sums
        rows.append(
            dict(
                date=str(day.date()),
                llc_mean=sums[1] / sums[0],
                obs_mean=sums[2] / sums[0],
                difference=sums[3] / sums[0],
                rmse=np.sqrt(sums[4] / sums[0]),
                observation_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    bias = totals[3] / totals[0]
    records.append(
        dict(
            variable=variable,
            observation=product,
            units=unit,
            mean_difference=bias,
            rmse=np.sqrt(totals[4] / totals[0]),
            centered_rmse=np.sqrt(max(0, totals[4] / totals[0] - bias**2)),
            llc_mean=totals[1] / totals[0],
            observation_mean=totals[2] / totals[0],
            dates=rows,
        )
    )
result = dict(
    description="32 evenly spaced training dates; midnight LLC snapshots versus same-date daily analyses, joint finite wet-area support. Temporal support differs. Bias is not an identified datum correction; no transformation applied.",
    grid_proof_sha256=hashlib.sha256(
        (args.prepared / "grid.json").read_bytes()
    ).hexdigest(),
    fields=records,
)
args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(
    json.dumps(
        [{k: v for k, v in record.items() if k != "dates"} for record in records]
    ),
    flush=True,
)
