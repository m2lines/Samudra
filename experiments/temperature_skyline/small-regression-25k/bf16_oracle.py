# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""CPU nearest-BF16 and 8192-center oracles on saved physical map targets.

Uses checkpoint normalization and affine FP32 physical readout. No inference or
training. Input maps contain already-rounded physical targets; sub-micro-unit
roundoff is distinct from the model and vocabulary errors assessed here.
"""

import argparse
import csv
import json
import pathlib

import numpy as np
import torch

from samudra.skyline.channel_vocabulary import ChannelVocabulary

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=pathlib.Path, required=True)
parser.add_argument("--output", type=pathlib.Path, required=True)
args = parser.parse_args()
root = args.root
receipt = json.loads((root / "new-regression-receipt.json").read_text())
src = receipt["source"]
v = ChannelVocabulary(receipt["config"]["vocabulary_bounds"], 8192)
rows = []
for p in sorted(root.glob("*.npz")):
    a = np.load(p)
    wet = a["wet"]
    w = a["area"][wet]
    depth = int(a["depth"])
    for field, start in [("uo", 0), ("vo", 19), ("thetao", 38), ("so", 57)]:
        if field + "_prediction" not in a:
            continue
        c = start + depth
        mean = np.float32(src["mean"][c])
        std = np.float32(src["std"][c])
        gold = a[field + "_gold"][wet].astype(np.float64)
        pred = a[field + "_prediction"][wet]
        z = (gold - float(mean)) / float(std)
        bf = torch.from_numpy(z).to(torch.bfloat16).float().numpy()
        bf_phys = (bf * std + mean).astype(np.float64)
        centers = v.centers[c].numpy().astype(np.float64) * float(std) + float(mean)
        oracle8192 = centers[np.searchsorted((centers[1:] + centers[:-1]) / 2, gold)]
        pred_z = (pred.astype(np.float64) - float(mean)) / float(std)
        pred_back = (
            torch.from_numpy(pred_z).to(torch.bfloat16).float().numpy() * std + mean
        )
        row = dict(
            task=p.stem.removeprefix("new-regression-"),
            field=field,
            units={"thetao": "degC", "so": "PSU", "uo": "m/s", "vo": "m/s"}[field],
            normalization_mean=float(mean),
            normalization_std=float(std),
            depth_m=src["depths_m"][depth],
            wet_cells=int(wet.sum()),
            regression_area_rmse=float(
                np.sqrt(np.sum(w * (pred - gold) ** 2) / w.sum())
            ),
            bf16_oracle_area_rmse=float(
                np.sqrt(np.sum(w * (bf_phys - gold) ** 2) / w.sum())
            ),
            bin8192_oracle_area_rmse=float(
                np.sqrt(np.sum(w * (oracle8192 - gold) ** 2) / w.sum())
            ),
            predictions_exactly_on_BF16_physical_lattice=int(np.sum(pred_back == pred)),
            prediction_lattice_max_abs_error=float(np.max(np.abs(pred_back - pred))),
        )
        row["bf16_floor_fraction_of_actual_mse"] = (
            row["bf16_oracle_area_rmse"] / row["regression_area_rmse"]
        ) ** 2
        rows.append(row)
r = args.output
r.mkdir(parents=True, exist_ok=True)
with (r / "bf16-oracle.csv").open("w") as f:
    wr = csv.DictWriter(f, fieldnames=rows[0])
    wr.writeheader()
    wr.writerows(rows)
for x in rows:
    print(
        x["task"],
        x["field"],
        "BF16oracle",
        x["bf16_oracle_area_rmse"],
        "reg",
        x["regression_area_rmse"],
        "fraction",
        x["bf16_floor_fraction_of_actual_mse"],
        "onlattice",
        x["predictions_exactly_on_BF16_physical_lattice"],
        x["wet_cells"],
    )
