# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Continuous annual evaluation with prescribed forcing and history-only surfaces."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from samudra.metrics import kernels, spectra
from samudra.observations.archive import ObservationArchive, context_planes
from samudra.observations.metrics import _field, spatial_error, weighted_rmse
from samudra.observations.prepare import save_atomic

LEADS = [5, 15, 30, 90, 180, 365]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_origin(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "COMPLETE.json").read_text())
    if (manifest["history_bins"], manifest["forecast_bins"], manifest["bin_days"]) != (
        19,
        73,
        5,
    ) or len(manifest["records"]) != 92:
        raise ValueError("Unexpected annual sequence layout")
    values = []
    for index, record in enumerate(manifest["records"]):
        path = directory / record["file"]
        if (
            Path(record["file"]).name != record["file"]
            or digest(path) != record["sha256"]
        ):
            raise ValueError("Annual payload path/hash mismatch")
        with np.load(path) as saved:
            item = dict(saved)
        expected = pd.Timestamp(manifest["origin"]) + pd.Timedelta(
            days=5 * (index - 19)
        )
        if (
            pd.Timestamp(str(item["start"])) != expected
            or pd.Timestamp(str(item["end"])) != expected + pd.Timedelta(days=5)
            or pd.Timestamp(str(item["midpoint"])) != expected + pd.Timedelta(days=2.5)
        ):
            raise ValueError("Annual timestamps are not contiguous exact five-day bins")
        values.append(item)
    return manifest, {
        key: np.stack([item[key] for item in values])
        for key in ("surface", "velocity", "atmosphere", "start", "midpoint")
    }


def inputs(data, raw):
    # Only history surface observations enter inference; future observations are targets.
    surface = raw["surface"][:19]
    validity = np.isfinite(surface) & data.grid["mask"][[38, 76]]
    months = pd.DatetimeIndex(raw["midpoint"][:19]).month.to_numpy() - 1
    filled = np.where(validity, surface, data.stats["surface_climatology"][months])
    normalized = (filled - data.grid["mean"][[38, 76], None, None]) / data.grid["std"][
        [38, 76], None, None
    ]
    normalized = np.where(validity, normalized, 0)
    normalized *= data.grid["mask"][[38, 76]]
    atmosphere = (
        raw["atmosphere"] - data.stats["atmosphere_mean"][None, :, None, None]
    ) / data.stats["atmosphere_std"][None, :, None, None]
    if not np.isfinite(normalized).all() or not np.isfinite(atmosphere).all():
        raise ValueError("Nonfinite annual model input")
    return (
        data.tensor(normalized)[None],
        data.tensor(atmosphere)[None],
        data.tensor(
            context_planes(data.grid["lat"], data.grid["lon"], raw["midpoint"])
        )[None],
        data.mask,
        data.tensor(validity)[None],
    )


def surface_metrics(prediction, reference, data):
    lat, lon = data.grid["lat"], data.grid["lon"]
    area = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, len(lon)))
    domain = data.grid["mask"][0].copy()
    p, r = np.where(domain, prediction, np.nan), np.where(domain, reference, np.nan)
    predicted_u, predicted_v = kernels.geostrophic_velocity_from_zos(
        _field(p[:, 1], lat, lon), "lat", "lon"
    )
    ru, rv = kernels.geostrophic_velocity_from_zos(
        _field(r[:, 1], lat, lon), "lat", "lon"
    )
    predicted_u, predicted_v, ru, rv = (
        field.transpose("time", "lat", "lon")
        for field in (predicted_u, predicted_v, ru, rv)
    )
    support = (
        (np.abs(lat[:, None]) >= 5)
        & np.isfinite(ru.values)
        & np.isfinite(rv.values)
        & np.isfinite(r[:, 2])
        & np.isfinite(r[:, 3])
    )
    pu, pv = (
        np.where(support, predicted_u.values, np.nan),
        np.where(support, predicted_v.values, np.nan),
    )
    u, v = np.where(support, r[:, 2], np.nan), np.where(support, r[:, 3], np.nan)
    result: dict[str, Any] = {"leads": {}, "spectra": {}}
    for day in LEADS:
        i = day // 5 - 1
        result["leads"][str(day)] = {
            "sst_rmse": weighted_rmse(p[i, 0], r[i, 0], area),
            "adt_rmse": weighted_rmse(p[i, 1], r[i, 1], area),
            "velocity_rmse": float(
                np.sqrt(2)
                * weighted_rmse(np.stack([pu[i], pv[i]]), np.stack([u[i], v[i]]), area)
            ),
        }
        for channel, label in enumerate(("sst", "adt")):
            for region in spectra.SPATIAL_REGIONS:
                curve = spatial_error(
                    p[i : i + 1, channel], r[i : i + 1, channel], lat, lon, region
                )
                if curve is not None:
                    result["spectra"][f"{label}/{region[0]}/day{day}"] = curve
    # Within-year temporal anomalies, not EKE across independently initialized origins.
    pe = kernels.instantaneous_surface_eke(
        _field(pu, lat, lon), _field(pv, lat, lon)
    ).values
    re = kernels.instantaneous_surface_eke(
        _field(u, lat, lon), _field(v, lat, lon)
    ).values
    result["annual_eke_rmse"] = weighted_rmse(pe, re, area)
    for region in spectra.SPATIAL_REGIONS:
        curve = spatial_error(pe, re, lat, lon, region)
        if curve is not None:
            result["spectra"][f"annual-eke/{region[0]}"] = curve
    return result


@torch.no_grad()
def evaluate_year(model, data, directory, output):
    manifest, raw = read_origin(directory)
    surface, atmosphere, contexts, mask, validity = inputs(data, raw)
    model.eval()
    predictions = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        states = model.initialize(
            surface, atmosphere[:, :19], contexts[:, 18], mask, validity
        )
        initial = states.float().cpu().numpy()
        forcing = model.adapt(atmosphere[:, 19:])
        for step in range(73):
            prediction = model.evolution(
                states, forcing[:, step : step + 1], contexts[:, 19 + step], mask, 1
            )
            predictions.append(
                data.physical(prediction[:, None])[0, 0, [38, 76]].cpu().numpy()
            )
            states = torch.stack((states[:, -1], prediction), 1)
    reference = np.concatenate((raw["surface"][19:], raw["velocity"][19:]), axis=1)
    result = surface_metrics(np.array(predictions), reference, data)
    result["origin"] = manifest["origin"]
    save_atomic(
        output, initial=initial, surface=np.array(predictions), reference=reference
    )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Prepare a continuous year from the daily archive"
    )
    parser.add_argument("--daily", type=Path, required=True)
    parser.add_argument("--origin", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    archive = ObservationArchive(args.daily)
    origin = pd.Timestamp(args.origin)
    records = []
    for i in range(92):
        start = origin + pd.Timedelta(days=5 * (i - 19))
        end = start + pd.Timedelta(days=5)
        adt = archive.interval("adt", start, end)
        path = args.output / f"{i:03d}.npz"
        save_atomic(
            path,
            surface=np.concatenate((archive.interval("sst", start, end), adt[:1])),
            velocity=adt[1:],
            atmosphere=archive.interval("era5", start, end),
            start=np.array(start.isoformat()),
            end=np.array(end.isoformat()),
            midpoint=np.array((start + pd.Timedelta(days=2.5)).isoformat()),
        )
        records.append({"file": path.name, "sha256": digest(path)})
    (args.output / "COMPLETE.json").write_text(
        json.dumps(
            {
                "origin": origin.isoformat(),
                "history_bins": 19,
                "forecast_bins": 73,
                "bin_days": 5,
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
