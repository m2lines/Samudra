#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Compare independently initialized December means using unchanged climatology."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from global_physical_heat_content import cell_areas  # type: ignore[import-not-found]
from plot_global_initializer_interior import (  # type: ignore[import-not-found]
    depth_axis,
    reduce_depth,
)

LABELS = {
    "obs08000": "Final: 8,000 obs",
    "scratch08000": "Obs-only: 8,000 obs",
    "scratch16000": "Obs-only: 16,000 obs",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--old-mixed", type=Path, required=True)
    parser.add_argument("--old-scratch", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = json.loads((args.arrays / "COMPLETE.json").read_text())
    for name, info in receipt["files"].items():
        if (
            digest(args.arrays / name) != info["sha256"]
            or (args.arrays / name).stat().st_size != info["bytes"]
        ):
            raise ValueError("Monthly array readback failed: " + name)
    grid = dict(np.load(args.arrays / "grid.npz"))
    lat, lon, depths, wet = (grid[key] for key in ["lat", "lon", "depths", "mask"])
    area = cell_areas(lat, lon)
    climate = np.load(args.arrays / "december-climatology.npz")["ts"]
    np.testing.assert_array_equal(
        climate, np.load(args.old_mixed / "december-climatology.npz")["ts"]
    )
    np.testing.assert_array_equal(
        climate, np.load(args.old_scratch / "december-climatology.npz")["ts"]
    )
    old_grid = dict(np.load(args.old_mixed / "grid.npz"))
    np.testing.assert_array_equal(lat, old_grid["lat"])
    np.testing.assert_array_equal(lon, old_grid["lon"])
    np.testing.assert_array_equal(wet, old_grid["mask"][:, :14])
    result: dict = {
        "months": {},
        "sources": {
            str(args.arrays / "COMPLETE.json"): digest(args.arrays / "COMPLETE.json")
        },
        "definition": receipt["definition"],
    }
    for month in ["2014-12", "2017-12", "2020-12"]:
        reference = np.load(args.arrays / (month + "-reference.npz"))["ts"]
        np.testing.assert_array_equal(
            reference, np.load(args.references / (month + ".npz"))["values"]
        )
        origin = str(int(month[:4]) + 1) + "-01-01"
        prediction = {
            s: np.load(args.arrays / (s + "-" + month + ".npz"))["ts"] for s in LABELS
        }
        old = {
            s: np.load(
                (args.old_mixed if s == "obs08000" else args.old_scratch)
                / (s + "-" + origin + ".npz")
            )["ts"][:, :14]
            for s in LABELS
        }
        for s in LABELS:
            path = (args.old_mixed if s == "obs08000" else args.old_scratch) / (
                s + "-" + origin + ".npz"
            )
            result["sources"][str(path)] = digest(path)
        support = wet & np.isfinite(reference) & np.isfinite(climate)
        if not all(
            np.isfinite(v[support]).all() for v in [*prediction.values(), *old.values()]
        ):
            raise ValueError("Nonfinite model values on common support")
        methods = {**prediction, "IAP monthly mean": reference}
        record: dict = {
            "agreement": {},
            "northern_temperature_10_105m_anomaly_C": {},
            "zonal_temperature_correlations": {},
        }
        for s, values in {
            **prediction,
            "Training December climatology": climate,
        }.items():
            error = np.sqrt(reduce_depth((values - reference) ** 2, support, area))
            record["agreement"][s] = {
                "temperature_rmse_mean_10_1850_C": float(error[0, 1:].mean()),
                "salinity_rmse_mean_2_5_1850_psu": float(error[1].mean()),
            }
        lev = (depths >= 10) & (depths <= 105)
        band = (lat >= 30) & (lat <= 60)
        weight = np.where(support[0, lev] & band[:, None], area, 0)
        for s, values in methods.items():
            record["northern_temperature_10_105m_anomaly_C"][s] = {
                "monthly_mean": float(
                    (
                        np.where(support[0, lev], values[0, lev] - climate[0, lev], 0)
                        * weight
                    ).sum()
                    / weight.sum()
                )
            }
            if s in old:
                record["northern_temperature_10_105m_anomaly_C"][s][
                    "last_pre_January_state"
                ] = float(
                    (
                        np.where(support[0, lev], old[s][0, lev] - climate[0, lev], 0)
                        * weight
                    ).sum()
                    / weight.sum()
                )
        sections = {}
        fig, axes = plt.subplots(2, 4, figsize=(14, 7), layout="constrained")
        for col, (method, values) in enumerate(methods.items()):
            weights = np.where(support, area, 0)
            section = np.divide(
                (np.where(support, values - climate, 0) * weights).sum(-1),
                weights.sum(-1),
                out=np.full(values.shape[:-1], np.nan),
                where=weights.sum(-1) > 0,
            )
            sections[method] = section
            for variable in [0, 1]:
                ax = axes[variable, col]
                start = 1 if variable == 0 else 0
                limit = 1.5 if variable == 0 else 0.5
                im = ax.pcolormesh(
                    lat,
                    depths[start:],
                    section[variable, start:],
                    vmin=-limit,
                    vmax=limit,
                    cmap="RdBu_r",
                    shading="auto",
                )
                depth_axis(ax)
                ax.set_xlabel("Latitude (°N)")
                ax.set_title(LABELS.get(method, method), fontsize=10)
                if col == 0:
                    ax.set_ylabel(
                        "Temperature depth (m)"
                        if variable == 0
                        else "Salinity depth (m)"
                    )
                if col == 3:
                    fig.colorbar(
                        im,
                        ax=list(axes[variable]),
                        label="Temperature anomaly (°C)"
                        if variable == 0
                        else "Salinity anomaly (psu)",
                        shrink=0.8,
                    )
        for stage in LABELS:
            x, y = sections[stage][0, 1:], sections["IAP monthly mean"][0, 1:]
            valid = np.isfinite(x) & np.isfinite(y)
            record["zonal_temperature_correlations"][stage] = float(
                np.corrcoef(x[valid], y[valid])[0, 1]
            )
        fig.suptitle(
            f"{month}: independently initialized whole-December means\nAnomalies from training December climatology (1993–2012); common observed support"
        )
        fig.savefig(args.output / (month + "-initializer-zonal-anomalies.png"), dpi=140)
        plt.close(fig)
        result["months"][month] = record
    result["files"] = {
        p.name: {"sha256": digest(p), "bytes": p.stat().st_size}
        for p in args.output.glob("*.png")
    }
    result["script_sha256"] = digest(__file__)
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                m: r["northern_temperature_10_105m_anomaly_C"]
                for m, r in result["months"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
