# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Score physical predictions on identical strict support, never silently drop NaNs."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import xarray as xr


def score(root, arm):
    preparation = json.loads((root / "preparation.json").read_text())
    original = xr.open_zarr(Path(preparation["data"]) / "OM4.zarr")
    stds = xr.open_zarr(Path(preparation["data"]) / "OM4_stds.zarr")
    geometry = np.load(root / "geometry/geometry.npz")
    masks, area = geometry["masks"], geometry["area"]
    rows = []
    for year in [2015, 2017, 2019, 2021]:
        prediction = xr.open_zarr(root / "eval" / f"{arm}-{year}" / "predictions.zarr")
        assert prediction.sizes["time"] == 72, prediction.sizes
        indices = original.indexes["time"].get_indexer(prediction.time.values)
        assert (indices >= 0).all() and np.all(np.diff(indices) == 1)
        initial_index = indices[0] - 1
        truth = original.isel(time=indices)
        np.testing.assert_allclose(prediction.x.values, original.x.values)
        np.testing.assert_allclose(prediction.y.values, original.y.values)
        for base in ["uo", "vo", "thetao", "so", "zos"]:
            levels = range(19) if base != "zos" else [0]
            for k in levels:
                name = f"{base}_{k}" if base != "zos" else base
                wet = masks[k]
                if not wet.any():
                    continue  # Documented empty depth, never report zero error.
                p = (
                    prediction[base].isel(lev=k).values
                    if base != "zos"
                    else prediction[base].values
                )
                t = truth[name].values
                persistence = original[name].isel(time=initial_index).values
                assert np.isfinite(p[:, wet]).all(), (
                    arm,
                    year,
                    name,
                    "invalid prediction",
                )
                assert np.isfinite(t[:, wet]).all(), (arm, year, name, "invalid truth")
                assert np.isfinite(persistence[wet]).all()
                w = area[wet].astype("float64")
                w /= w.sum()
                errors = p[:, wet].astype("float64") - t[:, wet]
                rmse = np.sqrt(np.sum(errors**2 * w, axis=1))
                bias = np.sum(errors * w, axis=1)
                persist_rmse = np.sqrt(
                    np.sum((persistence[wet] - t[:, wet]) ** 2 * w, axis=1)
                )
                std = float(stds[name].values.squeeze())
                assert np.isfinite(std) and std > 0
                for i in range(72):
                    delta = (
                        prediction.time.values[i] - original.time.values[initial_index]
                    )
                    lead = (
                        int(delta / np.timedelta64(1, "D"))
                        if isinstance(delta, np.timedelta64)
                        else delta.days
                    )
                    assert lead == (i + 1) * 5
                    rows.append(
                        dict(
                            arm=arm,
                            origin_year=year,
                            lead_days=lead,
                            variable=base,
                            level=k if base != "zos" else -1,
                            channel=name,
                            cells=int(wet.sum()),
                            rmse=float(rmse[i]),
                            normalized_rmse=float(rmse[i] / std),
                            bias=float(bias[i]),
                            persistence_rmse=float(persist_rmse[i]),
                            persistence_normalized_rmse=float(persist_rmse[i] / std),
                        )
                    )
    out = root / "scores"
    out.mkdir(exist_ok=True)
    with (out / f"{arm}.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(
        arm=arm,
        mean_normalized_rmse=float(np.mean([r["normalized_rmse"] for r in rows])),
        mean_persistence_normalized_rmse=float(
            np.mean([r["persistence_normalized_rmse"] for r in rows])
        ),
        empty_levels=np.where(~masks.any(axis=(1, 2)))[0].tolist(),
        metric="mean over 4 origins, 72 lead times and nonempty channels of spatial area-weighted RMSE / original training std; lower is better",
        primary_checkpoint="final epoch 70 EMA, not best validation checkpoint",
        scope="prepared OM4 model truth; one training seed",
    )
    (out / f"{arm}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


def compare(root):
    import matplotlib
    import pandas as pd

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = root / "scores"
    a = pd.read_csv(out / "baseline.csv")
    b = pd.read_csv(out / "nonpartial.csv")
    keys = ["origin_year", "lead_days", "variable", "level", "channel", "cells"]
    merged = a.merge(
        b, on=keys, suffixes=("_baseline", "_nonpartial"), validate="one_to_one"
    )
    assert len(merged) == len(a) == len(b)
    np.testing.assert_allclose(
        merged.persistence_rmse_baseline, merged.persistence_rmse_nonpartial
    )
    merged["relative_change_percent"] = 100 * (
        merged.normalized_rmse_nonpartial / merged.normalized_rmse_baseline - 1
    )
    merged.to_csv(out / "paired.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for arm, frame in [("baseline", a), ("nonpartial", b)]:
        curve = frame.groupby("lead_days").normalized_rmse.mean()
        axes[0].plot(curve.index, curve.values, label=arm)
    persistence = a.groupby("lead_days").persistence_normalized_rmse.mean()
    axes[0].plot(
        persistence.index, persistence.values, label="persistence", linestyle=":"
    )
    axes[0].set(
        xlabel="Forecast lead (days)", ylabel="Mean normalized RMSE (lower is better)"
    )
    axes[0].legend()
    byvar = merged.groupby("variable")[
        ["normalized_rmse_baseline", "normalized_rmse_nonpartial"]
    ].mean()
    changes = 100 * (
        byvar.normalized_rmse_nonpartial / byvar.normalized_rmse_baseline - 1
    )
    axes[1].bar(changes.index, changes.values)
    axes[1].axhline(0, color="black", lw=0.7)
    axes[1].set(
        ylabel="RMSE change (%) — negative improves",
        title="Identical fully wet support",
    )
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=170)
    paired = merged.groupby("origin_year")[
        ["normalized_rmse_baseline", "normalized_rmse_nonpartial"]
    ].mean()
    paired["relative_change_percent"] = 100 * (
        paired.normalized_rmse_nonpartial / paired.normalized_rmse_baseline - 1
    )
    paired.to_csv(out / "by-origin.csv")
    baseline = float(a.normalized_rmse.mean())
    treatment = float(b.normalized_rmse.mean())
    result = dict(
        baseline=baseline,
        nonpartial=treatment,
        relative_change_percent=100 * (treatment / baseline - 1),
        by_origin=paired.reset_index().to_dict(orient="records"),
        by_variable_percent=changes.to_dict(),
        protocol="four held-out 360-day origins; equal channel/time/origin average of area-weighted normalized RMSE; strict common support; final epoch EMA",
        caveat="single paired training seed; masks alter both boundary context and training objective, including dynamic channel weights; no inference about observational accuracy",
    )
    (out / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("arm", choices=["baseline", "nonpartial", "compare"])
    a = p.parse_args()
    compare(a.root) if a.arm == "compare" else score(a.root, a.arm)
