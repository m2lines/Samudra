# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Continuous-year diagnostics, separate from monthly checkpoint selection."""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from samudra.experiments.observation_data import context_planes
from samudra.experiments.observation_metrics import _field, spatial_error, weighted_rmse
from samudra.experiments.observation_model import ObservationTransfer
from samudra.experiments.observation_pilot import atomic_json, digest
from samudra.experiments.observation_training import Samples
from samudra.metrics import kernels, spectra

LEADS = [5, 15, 30, 90, 180, 365]


def month_weights(starts, month):
    first = pd.Period(month, freq="M").start_time
    last = first + pd.offsets.MonthBegin(1)
    overlaps = np.array(
        [
            max(pd.Timedelta(0), min(t + pd.Timedelta(days=5), last) - max(t, first))
            / pd.Timedelta(days=1)
            for t in pd.DatetimeIndex(starts)
        ]
    )
    if not np.isclose(overlaps.sum(), (last - first).days):
        raise ValueError(f"Incomplete calendar month: {month}")
    return overlaps / overlaps.sum()


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
    domain = data.grid["mask"][0] & (np.abs(lat[:, None]) <= 60)
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


def evaluate_origins(model, data, origins, output, signature):
    """Write the same year-long point diagnostics for any forecast interface.

    Ensemble callers must average independently evolved members before entering
    this interface; these point outputs do not measure ensemble calibration.
    """
    results = {}
    for origin_path in origins:
        description, raw = read_origin(origin_path.parent)
        origin = description["origin"]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            prediction, initial = model.forecast(*inputs(data, raw))
        if prediction.shape[1] != 73 or not torch.isfinite(prediction).all():
            raise ValueError(f"Nonfinite or incomplete annual rollout: {origin}")
        physical = data.physical(prediction)
        surface = physical[0, :, [38, 76]].cpu().numpy()
        reference = np.concatenate([raw["surface"][19:], raw["velocity"][19:]], axis=1)
        result = surface_metrics(surface, reference, data)
        months, monthly_predictions, monthly_references = [], [], []
        area = data.area.cpu().numpy()
        for record in description["monthly_interiors"]:
            path = origin_path.parent / record["file"]
            if (
                Path(record["file"]).name != record["file"]
                or digest(path) != record["sha256"]
            ):
                raise ValueError("Monthly target hash/path mismatch")
            month = path.name[:7]
            weights = data.tensor(month_weights(raw["start"][19:], month))
            state = (prediction * weights[None, :, None, None, None]).sum(1)
            ohc = data.ohc(state)[0]
            with np.load(path) as saved:
                truth = np.where(np.isfinite(ohc), saved["ohc"], np.nan)
            result.setdefault("monthly_ohc", {})[month] = {
                label: weighted_rmse(ohc[i], truth[i], area)
                for i, label in enumerate(("0_700", "700_2000"))
            }
            months.append(month)
            monthly_predictions.append(ohc)
            monthly_references.append(truth)
        result["scope"] = (
            "Single initialization; 73 five-day steps; no future surface corrections"
        )
        atomic_json(result, output / f"{origin}.json")
        np.savez_compressed(
            output / f"{origin}.npz",
            surface=surface,
            reference=reference,
            initial=data.physical(initial)[0].cpu().numpy(),
            state_at_leads=physical[0, [day // 5 - 1 for day in LEADS]].cpu().numpy(),
            leads_days=LEADS,
            months=months,
            predicted_ohc=np.array(monthly_predictions),
            reference_ohc=np.array(monthly_references),
        )
        results[origin] = {"metrics": f"{origin}.json", "arrays": f"{origin}.npz"}
        print(
            json.dumps({"event": "annual_origin_complete", "origin": origin}),
            flush=True,
        )
    atomic_json({"inputs": signature, "origins": results}, output / "COMPLETE.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--annual-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    run, annual, output = Path(args.run), Path(args.annual_data), Path(args.output)
    manifest = json.loads((run / "manifest.json").read_text())
    checkpoint = run / args.checkpoint
    if args.split == "test":
        if not (run / "TRAIN_COMPLETE.json").exists() or args.checkpoint != "best.pt":
            raise ValueError(
                "Held-out annual evaluation requires completed selected weights"
            )
        if (
            digest(checkpoint)
            != json.loads((run / "best.json").read_text())["checkpoint_sha256"]
        ):
            raise ValueError("Selected checkpoint hash differs")
    if not (annual / "DATA_READY.json").exists():
        raise ValueError("Annual data must pass full transfer verification")
    origins = sorted((annual / args.split).glob("*/COMPLETE.json"))
    if len(origins) != (1 if args.split == "validation" else 3):
        raise ValueError("Incomplete annual cohort")
    signature = {
        "protocol": "continuous-365-day-v1",
        "training_manifest_sha256": digest(run / "manifest.json"),
        "checkpoint_sha256": digest(checkpoint),
        "data_manifests": {str(p.relative_to(annual)): digest(p) for p in origins},
        "producer": os.environ.get("SAMUDRA_CODE_COMMIT"),
        "split": args.split,
        "selection": "Diagnostic only; nine-origin v3 selection unchanged",
        "eke": "Anomalies relative to each sequence's own year mean",
        "forcing": "Prescribed ERA5 through trained adapter; not an operational forcing forecast",
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "input.json").exists() and json.loads(
        (output / "input.json").read_text()
    ) != signature:
        raise ValueError("Annual evaluation resume contract differs")
    atomic_json(signature, output / "input.json")
    data = Samples(manifest["arguments"]["data"], "cuda")
    if manifest["normalization_mode"] == "observation-only":
        data.use_observation_normalization()
    model = (
        ObservationTransfer(
            data.grid["names"].tolist(),
            manifest["arguments"].get("normalization", "batch"),
        )
        .cuda()
        .eval()
    )
    model.load_state_dict(
        torch.load(checkpoint, map_location="cuda", weights_only=False)["model"],
        strict=True,
    )
    evaluate_origins(model, data, origins, output, signature)


if __name__ == "__main__":
    main()
