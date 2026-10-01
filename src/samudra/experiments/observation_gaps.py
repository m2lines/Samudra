# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen-checkpoint missing-surface interventions; no training or reselection."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

from samudra.experiments.observation_annual import inputs, read_origin, surface_metrics
from samudra.experiments.observation_model import ObservationTransfer
from samudra.experiments.observation_pilot import atomic_json, digest
from samudra.experiments.observation_training import Samples

MODES = ("baseline", "initial-zero", "history-zero", "initial-om4", "history-om4")


def prepare_climatology(root, output):
    """Only source-training dates; actual physical values, never held-out labels."""
    dataset = xr.open_zarr(Path(root) / "OM4.zarr", consolidated=True)
    dates = dataset.time.values
    indices = [
        i
        for i, date in enumerate(dates)
        if (1975, 1, 3) <= (date.year, date.month, date.day) <= (2013, 7, 31)
    ]
    if not indices:
        raise ValueError("No OM4 training dates")
    months = np.array([dates[i].month for i in indices])
    values, counts = [], []
    for month in range(1, 13):
        selected = np.array(indices)[months == month]
        means = []
        for variable in ("thetao_0", "zos"):
            means.append(
                dataset[variable]
                .isel(time=selected)
                .mean("time", skipna=True)
                .compute(scheduler="threads", num_workers=4)
                .values
            )
        values.append(np.stack(means))
        counts.append(len(selected))
        print(
            json.dumps(
                {
                    "event": "om4_climatology_month",
                    "month": month,
                    "frames": len(selected),
                }
            ),
            flush=True,
        )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        surface=np.array(values, dtype=np.float32),
        count=counts,
        lat=dataset.y.values,
        lon=dataset.x.values,
    )
    atomic_json(
        {
            "source": str(Path(root) / "OM4.zarr"),
            "source_metadata_sha256": digest(Path(root) / "OM4.zarr/.zmetadata"),
            "training_start": str(dates[indices[0]]),
            "training_end": str(dates[indices[-1]]),
            "monthly_counts": counts,
            "sha256": digest(output),
            "producer": os.environ.get("SAMUDRA_CODE_COMMIT"),
        },
        output.with_suffix(".json"),
    )


def replace_missing(values, validity, replacement, wet):
    if values.shape != validity.shape or replacement.shape != values.shape:
        raise ValueError("Missing-cell replacement shape mismatch")
    support = ~validity.bool() & wet.bool() & torch.isfinite(replacement)
    result = torch.where(support, replacement, values)
    return result, support


def forecast(model, arguments, replacements, mode):
    if mode not in MODES:
        raise ValueError(mode)
    surface, atmosphere, contexts, mask, validity = arguments
    observed_surface = surface.clone()
    if mode.startswith("history-"):
        surface, _ = replace_missing(surface, validity, replacements, mask[[38, 76]])
    initial = model.initialize(
        surface, atmosphere[:, :19], contexts[:, 18], mask, validity
    )
    if mode.startswith("initial-"):
        initial = initial.clone()
        changed, _ = replace_missing(
            initial[:, :, [38, 76]],
            validity[:, -2:],
            replacements[:, -2:],
            mask[[38, 76]],
        )
        initial[:, :, [38, 76]] = changed
    if not torch.equal(
        initial[:, :, [38, 76]][validity[:, -2:].bool()],
        observed_surface[:, -2:][validity[:, -2:].bool()],
    ):
        raise ValueError("Intervention changed known initial observations")
    states = initial
    forcing = model.adapt(atmosphere[:, 19:])
    predictions = []
    for step in range(forcing.shape[1]):
        prediction = model.evolution(
            states, forcing[:, step : step + 1], contexts[:, 19 + step], mask, 1
        )
        predictions.append(prediction)
        states = torch.stack((states[:, -1], prediction), 1)
    return torch.stack(predictions, 1), initial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-om4-climatology", action="store_true")
    parser.add_argument("--om4-data", default="/scratch/jr7309/data/om4_onedeg_v3")
    parser.add_argument("--climatology")
    parser.add_argument("--run")
    parser.add_argument("--annual-data")
    parser.add_argument("--baseline-output")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.prepare_om4_climatology:
        prepare_climatology(args.om4_data, args.output)
        return
    if not all((args.run, args.annual_data, args.climatology, args.baseline_output)):
        parser.error(
            "Gap evaluation requires run, annual data, climatology and existing baseline outputs"
        )
    run, output = Path(args.run), Path(args.output)
    manifest = json.loads((run / "manifest.json").read_text())
    complete = json.loads((run / "TRAIN_COMPLETE.json").read_text())
    checkpoint_hash = digest(run / "best.pt")
    if checkpoint_hash != complete["best_checkpoint_sha256"]:
        raise ValueError("Selected model differs from completed training")
    data = Samples(manifest["arguments"]["data"], "cuda")
    if manifest["normalization_mode"] != "observation-only":
        raise ValueError("This diagnostic requires the completed shared-scaling models")
    data.use_observation_normalization()
    model = (
        ObservationTransfer.from_arguments(
            data.grid["names"].tolist(), manifest["arguments"]
        )
        .cuda()
        .eval()
    )
    model.load_state_dict(
        torch.load(run / "best.pt", map_location="cuda", weights_only=False)["model"],
        strict=True,
    )
    with np.load(args.climatology) as saved:
        climatology = saved["surface"]
        np.testing.assert_allclose(saved["lat"], data.grid["lat"], atol=1e-5, rtol=0)
        np.testing.assert_allclose(saved["lon"], data.grid["lon"], atol=1e-5, rtol=0)
    climo_record = json.loads(Path(args.climatology).with_suffix(".json").read_text())
    if climo_record["sha256"] != digest(args.climatology):
        raise ValueError("Climatology checksum mismatch")
    signature = {
        "checkpoint_sha256": checkpoint_hash,
        "manifest_sha256": digest(run / "manifest.json"),
        "climatology": climo_record,
        "modes": MODES,
        "producer": os.environ.get("SAMUDRA_CODE_COMMIT"),
        "annual_data": args.annual_data,
        "baseline_output": args.baseline_output,
        "scope": "Missing inputs only; OM4 climatology is simulation-informed even for scratch; no model selection",
    }
    output.mkdir(parents=True, exist_ok=True)
    signature["modes"] = list(MODES)
    if (output / "input.json").exists() and json.loads(
        (output / "input.json").read_text()
    ) != signature:
        raise ValueError("Gap diagnostic resume contract differs")
    atomic_json(signature, output / "input.json")
    if (output / "COMPLETE.json").exists():
        return
    origins = sorted((Path(args.annual_data) / "test").glob("*/COMPLETE.json"))
    if len(origins) != 3:
        raise ValueError("Require all three existing annual cases")
    results: dict[str, dict[str, str]] = {}
    for origin_path in origins:
        description, raw = read_origin(origin_path.parent)
        origin = description["origin"]
        arguments = inputs(data, raw)
        months = pd.DatetimeIndex(raw["midpoint"][:19]).month.to_numpy() - 1
        normalized = (
            climatology[months] - data.grid["mean"][[38, 76], None, None]
        ) / data.grid["std"][[38, 76], None, None]
        replacements = {
            "zero": torch.zeros_like(arguments[0]),
            "om4": data.tensor(normalized)[None],
        }
        reference = np.concatenate([raw["surface"][19:], raw["velocity"][19:]], 1)
        results[origin] = {}
        for mode in MODES:
            replacement = replacements["om4" if mode.endswith("om4") else "zero"]
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                prediction, initial = forecast(model, arguments, replacement, mode)
            physical = data.physical(prediction)
            surface = physical[0, :, [38, 76]].cpu().numpy()
            if not np.isfinite(surface).all():
                raise ValueError("Nonfinite gap intervention forecast")
            metrics = surface_metrics(surface, reference, data)
            if mode == "baseline":
                with np.load(
                    Path(args.baseline_output) / (origin + ".npz")
                ) as original:
                    difference = np.abs(surface - original["surface"])
                    # Cross-hardware BF16 equivalence is reported, never assumed exact.
                    metrics["baseline_max_absolute_difference"] = float(
                        difference.max()
                    )
                    metrics["baseline_rmse_difference"] = float(
                        np.sqrt(np.mean(difference.astype(np.float64) ** 2))
                    )
                    np.testing.assert_allclose(
                        surface, original["surface"], atol=1e-4, rtol=1e-5
                    )
            valid = np.isfinite(reference[:, :2]) & data.grid["mask"][[38, 76]]
            squared = np.where(valid, (surface - reference[:, :2]) ** 2, 0).sum(-1)
            count = valid.sum(-1)
            metrics["latitude_rmse"] = np.sqrt(
                np.divide(squared, count, out=np.zeros_like(squared), where=count > 0)
            ).tolist()
            metrics["latitude_support"] = count.tolist()
            atomic_json(metrics, output / (origin + "-" + mode + ".json"))
            np.savez_compressed(
                output / (origin + "-" + mode + ".npz"),
                initial=data.physical(initial)[0, :, [38, 47, 0, 76]].cpu().numpy(),
                surface=surface[[0, 5, 17, 35, 72]],
                reference=reference[[0, 5, 17, 35, 72], :2],
                validity=arguments[4][0].cpu().numpy(),
                input_surface=data.physical(initial)[0, :, [38, 76]].cpu().numpy(),
                lat=data.grid["lat"],
                lon=data.grid["lon"],
                leads_days=[5, 30, 90, 180, 365],
            )
            results[origin][mode] = origin + "-" + mode + ".json"
            print(
                json.dumps(
                    {"event": "gap_mode_complete", "origin": origin, "mode": mode}
                ),
                flush=True,
            )
    atomic_json({"inputs": signature, "results": results}, output / "COMPLETE.json")


if __name__ == "__main__":
    main()
