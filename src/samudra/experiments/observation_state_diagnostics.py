# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint state maps/profiles and causal-use diagnostics on validation only."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn

from samudra.experiments.observation_metrics import selection_score
from samudra.experiments.observation_model import ObservationTransfer
from samudra.experiments.observation_pilot import Pilot, atomic_json, digest
from samudra.experiments.observation_training import Samples


def intervene(initial, mode):
    if mode == "none":
        return initial
    modified = initial.clone()
    if mode == "zero-velocity":
        modified[:, :, :38] = 0
    elif mode == "mean-deep-ts":
        modified[:, :, 52:57] = 0
        modified[:, :, 71:76] = 0
    else:
        raise ValueError(mode)
    return modified


class StateIntervention(nn.Module):
    def __init__(self, model, mode):
        super().__init__()
        self.model = model
        self.mode = mode

    def forecast(self, surface, atmosphere, contexts, mask, validity):
        initial = self.model.initialize(
            surface[:, :19], atmosphere[:, :19], contexts[:, 18], mask, validity[:, :19]
        )
        initial = intervene(initial, self.mode)
        states = initial
        forcing = self.model.adapt(atmosphere[:, 19:])
        predictions = []
        for step in range(forcing.shape[1]):
            prediction = self.model.call(
                self.model.evolution,
                states,
                forcing[:, step : step + 1],
                contexts[:, 19 + step],
                mask,
                1,
            )
            predictions.append(prediction)
            states = torch.stack((states[:, -1], prediction), 1)
        return torch.stack(predictions, 1), initial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    args = parser.parse_args()
    run, output = Path(args.run), Path(args.output)
    manifest = json.loads((run / "manifest.json").read_text())
    if (
        manifest["normalization_mode"] != "observation-only"
        or manifest["arguments"]["normalization"] != "instance"
    ):
        raise ValueError(
            "This diagnostic requires common observation scaling and InstanceNorm"
        )
    reference = json.loads((run / "selection-reference.json").read_text())
    data = Samples(manifest["arguments"]["data"], "cuda")
    data.use_observation_normalization()
    model = ObservationTransfer(data.grid["names"].tolist(), "instance").cuda().eval()
    pilot: Any = Pilot.__new__(Pilot)
    pilot.data = data
    pilot.args = SimpleNamespace(strict_velocity_support=True)
    paths = data.paths("validation")
    if len(paths) != 9:
        raise ValueError("Incomplete validation cohort")
    output.mkdir(parents=True, exist_ok=True)
    signature: dict[str, Any] = {
        "manifest_sha256": digest(run / "manifest.json"),
        "checkpoints": {name: digest(run / name) for name in args.checkpoints},
        "producer": os.environ.get("SAMUDRA_CODE_COMMIT"),
        "validation_origins": [p.stem for p in paths],
        "scope": "Diagnostic only; interventions demonstrate forecast dependence, not physical identifiability. Monthly reconstruction uses observed surfaces throughout the target month.",
    }
    if (output / "input.json").exists() and json.loads(
        (output / "input.json").read_text()
    ) != signature:
        raise ValueError("Diagnostic resume contract differs")
    atomic_json(signature, output / "input.json")
    with torch.inference_mode():
        for name in args.checkpoints:
            model.load_state_dict(
                torch.load(run / name, map_location="cuda", weights_only=False)[
                    "model"
                ],
                strict=True,
            )
            pilot.model = model
            baseline = pilot.evaluate(paths)
            score = selection_score(
                baseline, reference["control"], reference["spectral_keys"]
            )
            interventions = {}
            for mode in ("zero-velocity", "mean-deep-ts"):
                pilot.model = StateIntervention(model, mode).eval()
                metrics = pilot.evaluate(paths)
                value = selection_score(
                    metrics, reference["control"], reference["spectral_keys"]
                )
                interventions[mode] = {
                    "score": value,
                    "score_change": value - score,
                    "metrics": metrics,
                }
            pilot.model = model
            records, initial_maps, reconstruction_maps, forecast_maps = [], [], [], []
            weights = data.area.cpu().numpy() * data.grid["mask"]
            denominator = weights.sum((-2, -1))
            for i, path in enumerate(paths):
                sample = data.load(path)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    forecast, initial = model.forecast(*Pilot.arguments(pilot, sample))
                    reconstruction = model.reconstruct_month(
                        *Pilot.arguments(pilot, sample), sample["month_weights"]
                    )
                physical = data.physical(initial)[0, -1].cpu().numpy()
                monthly = data.physical(reconstruction[:, None])[0, 0].cpu().numpy()
                target = sample["raw"]["interior"].reshape(28, *physical.shape[-2:])
                valid = np.isfinite(target) & data.ts_mask.cpu().numpy().astype(bool)
                support = valid * data.area.cpu().numpy()
                count = support.sum((-2, -1))
                error = np.where(valid, monthly[data.ts_indices] - target, 0)
                mse = (error**2 * support).sum((-2, -1)) / np.maximum(count, 1e-12)
                records.append(
                    {
                        "origin": path.stem,
                        "initial_mean": (
                            (physical * weights).sum((-2, -1)) / denominator
                        ).tolist(),
                        "initial_rms": np.sqrt(
                            (physical**2 * weights).sum((-2, -1)) / denominator
                        ).tolist(),
                        "monthly_reconstruction_mse": [
                            float(v) if c > 0 else None
                            for v, c in zip(mse, count, strict=True)
                        ],
                    }
                )
                if i in (0, len(paths) - 1):
                    initial_maps.append(physical)
                    reconstruction_maps.append(monthly)
                    forecast_maps.append(data.physical(forecast)[0, 5].cpu().numpy())
            stem = Path(name).stem
            atomic_json(
                {
                    "checkpoint_sha256": signature["checkpoints"][name],
                    "baseline_score": score,
                    "baseline": baseline,
                    "interventions": interventions,
                    "profiles": records,
                },
                output / f"{stem}.json",
            )
            np.savez_compressed(
                output / f"{stem}.npz",
                lat=data.grid["lat"],
                lon=data.grid["lon"],
                mask=data.grid["mask"],
                names=data.grid["names"],
                cases=[paths[0].stem, paths[-1].stem],
                initial=np.array(initial_maps),
                monthly_reconstruction=np.array(reconstruction_maps),
                day30=np.array(forecast_maps),
            )
            print(
                json.dumps({"event": "state_diagnostic_complete", "checkpoint": name}),
                flush=True,
            )
    atomic_json(signature, output / "COMPLETE.json")


if __name__ == "__main__":
    main()
