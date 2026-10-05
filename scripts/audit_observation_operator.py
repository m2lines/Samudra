#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic only: score contemporaneous observed ADT through the model operator."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from samudra.experiments.observation_metrics import PROTOCOL, score
from samudra.metrics import kernels


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def velocity_support_audit(prediction, reference, grid):
    """Partition common scoring support by availability of observed ADT stencils."""
    lat, lon = grid["lat"], grid["lon"]
    domain = grid["mask"][0] & (np.abs(lat[:, None]) <= 60)
    area = np.cos(np.deg2rad(lat))[:, None]

    def velocity(adt):
        field = xr.DataArray(
            np.where(domain, adt, np.nan),
            dims=("time", "lat", "lon"),
            coords={"lat": lat, "lon": lon},
        )
        return [
            value.transpose("time", "lat", "lon").values
            for value in kernels.geostrophic_velocity_from_zos(field, "lat", "lon")
        ]

    output = {}
    for lead in (0, 2, 5):
        pu, pv = velocity(prediction[:, lead, 1])
        ou, ov = velocity(reference[:, lead, 1])
        u, v = reference[:, lead, 2], reference[:, lead, 3]
        reference_velocity = np.stack([u, v], 1)
        predicted_velocity = np.stack([pu, pv], 1)
        valid = (
            np.isfinite(reference_velocity)
            & (np.isfinite(pu).all(0) & np.isfinite(pv).all(0))[None, None]
            & (np.abs(lat[:, None]) >= 5)
        )
        observed_stencil = (np.isfinite(ou) & np.isfinite(ov))[:, None]
        strict = np.stack([ou, ov], 1)
        if not np.allclose(
            predicted_velocity[valid & observed_stencil],
            strict[valid & observed_stencil],
        ):
            raise ValueError("Filling changed a complete observed stencil")
        groups: dict[str, Any] = {
            "reference_component_support_equal": bool(
                np.array_equal(np.isfinite(u), np.isfinite(v))
            )
        }
        total_weight = np.sum(valid * area)
        for name, accepted in (
            ("all_scored", valid),
            ("complete_observed_stencil", valid & observed_stencil),
            ("incomplete_observed_stencil", valid & ~observed_stencil),
        ):
            mass = np.sum(accepted * area)
            groups[name] = {
                "component_area_origin_fraction": float(mass / total_weight),
                "vector_rmse_m_s": float(
                    np.sqrt(
                        2
                        * np.sum(
                            np.where(
                                accepted,
                                (predicted_velocity - reference_velocity) ** 2,
                                0,
                            )
                            * area
                        )
                        / mass
                    )
                )
                if mass > 0
                else None,
                "reference_vector_rms_m_s": float(
                    np.sqrt(
                        2
                        * np.sum(np.where(accepted, reference_velocity**2, 0) * area)
                        / mass
                    )
                )
                if mass > 0
                else None,
            }
        output[f"day{5 * (lead + 1)}"] = groups
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--climatology-export", required=True, type=Path)
    parser.add_argument("--grid", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with np.load(args.climatology_export) as archive:
        arrays = {key: archive[key] for key in archive.files}
    with np.load(args.grid) as archive:
        grid = {key: archive[key] for key in archive.files}
    expected = [f"{y}-{m:02d}" for y in range(2015, 2023) for m in range(1, 13)]
    if arrays["origins"].tolist() != expected:
        raise ValueError("Expected the complete held-out cohort")
    surface = arrays["reference"][:, :, :2]
    # Actual target-time observations, not persisted initial observations. Fill
    # only unavailable targets to keep the operator's surrounding model domain.
    prediction = np.where(np.isfinite(surface), surface, arrays["prediction"])
    result = score(
        prediction=prediction,
        reference=arrays["reference"],
        predicted_ohc=arrays["reference_ohc"],
        reference_ohc=arrays["reference_ohc"],
        lat=grid["lat"],
        lon=grid["lon"],
        mask=grid["mask"][0],
    )
    if len(result["spectra"]) != 27:
        raise ValueError("Missing spectral components")
    for name in ("sst_rmse", "ohc_0_700_rmse", "ohc_700_2000_rmse"):
        if result["metrics"][name] != 0:
            raise ValueError("Observation self-comparison must have zero direct error")
    for key, curve in result["spectra"].items():
        if not key.startswith("eke/") and abs(curve["error_dex"]) > 1e-12:
            raise ValueError("Direct observed SST/ADT spectra must agree")
    support = velocity_support_audit(prediction, arrays["reference"], grid)
    for lead, groups in support.items():
        if not np.isclose(
            groups["all_scored"]["vector_rmse_m_s"],
            result["metrics"][f"velocity_rmse/{lead}"],
            rtol=1e-7,
        ):
            raise ValueError("Partitioned audit must reproduce the unchanged scorer")
    output = {
        "completed_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "scope": "Data-only operator diagnostic, not a forecast, candidate or error floor",
        "method": "Contemporaneous target SST/ADT on the common coarse grid, training climatology only at missing cells, then unchanged geostrophic and EKE scoring operators against directly coarsened DUACS velocity",
        "origins": expected,
        "climatology_export_sha256": digest(args.climatology_export),
        "grid_sha256": digest(args.grid),
        "script_sha256": digest(Path(__file__)),
        "velocity_support_audit": support,
        "protocol": PROTOCOL,
        "result": result,
    }
    with args.output.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
