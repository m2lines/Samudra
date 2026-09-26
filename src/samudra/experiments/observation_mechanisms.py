# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Training-derived state and forcing interventions on fixed annual checkpoints."""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from samudra.experiments.observation_annual import inputs, read_origin, surface_metrics
from samudra.experiments.observation_model import ObservationTransfer
from samudra.experiments.observation_pilot import atomic_json, digest
from samudra.experiments.observation_training import Samples

MODES = (
    "baseline",
    "initial-hidden-seasonal",
    "initial-velocity-seasonal",
    "day30-hidden-seasonal",
    "seasonal-future-forcing",
)


def replace_hidden(states, replacement, velocity_only=False):
    """Retain both observed-surface slots and use this model's own latent basis."""
    if states.shape != replacement.shape:
        raise ValueError("Seasonal state shape differs")
    result = states.clone()
    if velocity_only:
        result[:, :, :38] = replacement[:, :, :38]
    else:
        result[:] = replacement
        result[:, :, [38, 76]] = states[:, :, [38, 76]]
    return result


def training_means(model, data):
    paths = data.paths("train")
    if len(paths) != 243:
        raise ValueError("Incomplete training cohort for seasonal intervention")
    h, w = data.mask.shape[-2:]
    state_sum = np.zeros((12, 2, 77, h, w), np.float64)
    state_count = np.zeros(12, int)
    forcing_sum = np.zeros((12, 8, h, w), np.float64)
    forcing_count = np.zeros(12, int)
    seen = set()
    with torch.inference_mode():
        for index, path in enumerate(paths):
            sample = data.load(path)
            dates = pd.DatetimeIndex(sample["raw"]["midpoints"])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                state = model.initialize(
                    sample["surface"][:, :19],
                    sample["atmosphere"][:, :19],
                    sample["contexts"][:, 18],
                    data.mask,
                    sample["validity"][:, :19],
                )
            if not torch.isfinite(state).all():
                raise ValueError("Nonfinite training initializer state")
            month = dates[18].month - 1
            state_sum[month] += state[0].float().cpu().numpy()
            state_count[month] += 1
            forcing = sample["atmosphere"][0].cpu().numpy()
            for date, frame in zip(dates, forcing, strict=True):
                if date not in seen:
                    seen.add(date)
                    forcing_sum[date.month - 1] += frame
                    forcing_count[date.month - 1] += 1
            if index % 25 == 0:
                print(
                    json.dumps(
                        {
                            "event": "training_seasonal_state",
                            "completed": index + 1,
                            "total": len(paths),
                        }
                    ),
                    flush=True,
                )
    if (state_count == 0).any() or (forcing_count == 0).any():
        raise ValueError("Missing training calendar month")
    return {
        "state_mean": (state_sum / state_count[:, None, None, None, None]).astype(
            np.float32
        ),
        "forcing_mean": (forcing_sum / forcing_count[:, None, None, None]).astype(
            np.float32
        ),
        "state_count": state_count,
        "forcing_count": forcing_count,
    }


def forecast(model, arguments, months, means, mode):
    if mode not in MODES:
        raise ValueError(mode)
    surface, atmosphere, contexts, mask, validity = arguments
    initial = model.initialize(
        surface[:, :19], atmosphere[:, :19], contexts[:, 18], mask, validity[:, :19]
    )
    states = initial
    if mode == "seasonal-future-forcing":
        atmosphere = atmosphere.clone()
        atmosphere[:, 19:] = torch.as_tensor(
            means["forcing_mean"][months[19:]],
            device=atmosphere.device,
            dtype=atmosphere.dtype,
        )[None]
    forcing = model.adapt(atmosphere[:, 19:])
    predictions = []
    for step in range(forcing.shape[1]):
        intervene = (
            step == 0
            and mode in ("initial-hidden-seasonal", "initial-velocity-seasonal")
        ) or (step == 6 and mode == "day30-hidden-seasonal")
        if intervene:
            replacement = torch.as_tensor(
                means["state_mean"][months[18 + step]],
                device=states.device,
                dtype=states.dtype,
            )[None]
            states = replace_hidden(
                states, replacement, mode == "initial-velocity-seasonal"
            )
        prediction = model.call(
            model.evolution,
            states,
            forcing[:, step : step + 1],
            contexts[:, 19 + step],
            mask,
            1,
        )
        predictions.append(prediction)
        states = torch.stack((states[:, -1], prediction), 1)
    return torch.stack(predictions, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--annual-data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run, annual, output = Path(args.run), Path(args.annual_data), Path(args.output)
    manifest = json.loads((run / "manifest.json").read_text())
    if (
        not (run / "TRAIN_COMPLETE.json").exists()
        or manifest["arguments"].get("normalization") != "instance"
        or manifest["normalization_mode"] != "observation-only"
    ):
        raise ValueError(
            "Requires completed InstanceNorm model with shared observation scaling"
        )
    checkpoint = run / "best.pt"
    if (
        digest(checkpoint)
        != json.loads((run / "best.json").read_text())["checkpoint_sha256"]
    ):
        raise ValueError("Selected checkpoint hash differs")
    if not (annual / "DATA_READY.json").exists():
        raise ValueError("Annual data not verified")
    origins = sorted((annual / "test").glob("*/COMPLETE.json"))
    if len(origins) != 3:
        raise ValueError("Incomplete annual cohort")
    output.mkdir(parents=True, exist_ok=True)
    signature = {
        "protocol": "training-seasonal-state-and-forcing-interventions-v1",
        "producer": os.environ.get("SAMUDRA_CODE_COMMIT"),
        "checkpoint_sha256": digest(checkpoint),
        "training_manifest_sha256": digest(run / "manifest.json"),
        "training_statistics_sha256": digest(
            Path(manifest["arguments"]["data"]) / "statistics.npz"
        ),
        "data_manifests": {str(p.relative_to(annual)): digest(p) for p in origins},
        "modes": list(MODES),
        "scope": "Exploratory diagnostics on previously examined annual origins; no checkpoint selection or tuning. State replacements are this model's training-derived seasonal means; latent channels need not be physical quantities. Late replacement can be out of the evolved-state distribution.",
    }
    contract = output / "input.json"
    if contract.exists() and json.loads(contract.read_text()) != signature:
        raise ValueError("Mechanism evaluation resume differs")
    atomic_json(signature, contract)
    if (output / "COMPLETE.json").exists():
        return
    data = Samples(manifest["arguments"]["data"], "cuda")
    data.use_observation_normalization()
    model = (
        ObservationTransfer(
            data.grid["names"].tolist(),
            "instance",
            manifest["arguments"].get("evolution_architecture", "d"),
        )
        .cuda()
        .eval()
    )
    model.load_state_dict(
        torch.load(checkpoint, map_location="cuda", weights_only=False)["model"],
        strict=True,
    )
    mean_path = output / "training-means.npz"
    if mean_path.exists():
        with np.load(mean_path) as saved:
            means = dict(saved)
    else:
        means = training_means(model, data)
        temporary = mean_path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **means)
        temporary.replace(mean_path)
    results: dict[str, Any] = {}
    with torch.inference_mode():
        for origin_path in origins:
            description, raw = read_origin(origin_path.parent)
            origin = description["origin"]
            arguments = inputs(data, raw)
            months = pd.DatetimeIndex(raw["midpoint"]).month.to_numpy() - 1
            reference = np.concatenate(
                [raw["surface"][19:], raw["velocity"][19:]], axis=1
            )
            results[origin] = {}
            for mode in MODES:
                record_path = output / f"{origin}-{mode}.json"
                if record_path.exists():
                    results[origin][mode] = json.loads(record_path.read_text())
                    continue
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction = forecast(model, arguments, months, means, mode)
                    if mode == "baseline":
                        native, _ = model.forecast(*arguments)
                        torch.testing.assert_close(prediction, native, rtol=0, atol=0)
                        del native
                if prediction.shape[1] != 73 or not torch.isfinite(prediction).all():
                    raise ValueError(f"Nonfinite/incomplete {mode}: {origin}")
                surface = data.physical(prediction)[0, :, [38, 76]].cpu().numpy()
                metrics = surface_metrics(surface, reference, data)
                np.savez_compressed(output / f"{origin}-{mode}.npz", prediction=surface)
                atomic_json(metrics, record_path)
                results[origin][mode] = metrics
                print(
                    json.dumps(
                        {
                            "event": "mechanism_evaluated",
                            "origin": origin,
                            "mode": mode,
                            "leads": metrics["leads"],
                        }
                    ),
                    flush=True,
                )
    atomic_json(
        {
            "results": results,
            "training_means_sha256": digest(mean_path),
            "input": signature,
        },
        output / "COMPLETE.json",
    )


if __name__ == "__main__":
    main()
