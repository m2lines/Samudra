# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Global validation/test scores with member-level exports from one inference pass."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from samudra.experiments.diffusion_calibration import observation_ensemble_statistics
from samudra.experiments.diffusion_evaluation import evaluate_point_metrics
from samudra.experiments.diffusion_global import make_model
from samudra.experiments.diffusion_spatial import forecast_crps
from samudra.experiments.observation_pilot import atomic_json, digest
from samudra.experiments.observation_training import Samples


def serial(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    return value


class RecordingForecast(nn.Module):
    stochastic = True

    def __init__(self, model, data, paths, output):
        super().__init__()
        self.model, self.data, self.paths, self.output = model, data, paths, output
        self.records = []

    def forecast(self, *args, **kwargs):
        members, initial = self.model.forecast(*args, **kwargs)
        sample = self.data.load(self.paths[len(self.records)])
        stats = observation_ensemble_statistics(self.data, members, sample)
        stats["origin"] = sample["name"]
        stats["spatial_objective"] = float(forecast_crps(self.data, members, sample))
        # Gradients at decoded fields quantify the actual supervision strength;
        # they are not parameter-gradient norms or a fitting intervention.
        with torch.enable_grad():
            probe = members.detach().requires_grad_(True)
            pixel = forecast_crps(self.data, probe, sample, coefficient=0)
            spatial = forecast_crps(self.data, probe, sample) - pixel
            pixel_gradient = torch.autograd.grad(pixel, probe, retain_graph=True)[0]
            spatial_gradient = torch.autograd.grad(spatial, probe)[0]
            stats["loss_components"] = dict(
                pixel=float(pixel.detach()),
                spatial=float(spatial.detach()),
                pixel_field_gradient_norm=float(pixel_gradient.norm()),
                spatial_field_gradient_norm=float(spatial_gradient.norm()),
            )

        stats = serial(stats)
        self.records.append(stats)
        channels = [38, 76, 47, 66]
        physical = members.float() * self.data.std + self.data.mean
        monthly = (
            physical * sample["month_weights"][None, None, :, None, None, None]
        ).sum(2)
        np.savez_compressed(
            self.output / f"members-{sample['name']}.npz",
            day30=physical[:, 0, 5, channels].cpu().numpy(),
            initial=(initial.float() * self.data.std + self.data.mean)[
                :, 0, -1, channels
            ]
            .cpu()
            .numpy(),
            monthly=monthly[:, 0, self.data.ts_indices].cpu().numpy(),
            monthly_reference=sample["interior"][0].cpu().numpy(),
            day30_surface_reference=sample["raw_surface"][0, 5].cpu().numpy(),
            mask=self.data.grid["mask"][channels],
            lat=self.data.grid["lat"],
            lon=self.data.grid["lon"],
            channel_names=np.array(self.data.grid["names"])[channels],
            target_midpoint=sample["raw"]["midpoints"][24],
        )
        atomic_json(self.records, self.output / "calibration.json")
        print(
            json.dumps(
                dict(
                    event="origin_complete",
                    origin=sample["name"],
                    completed=len(self.records),
                    total=len(self.paths),
                )
            ),
            flush=True,
        )
        return members, initial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = Samples(args.data, "cuda", surface_fill="zero", global_observations=True)
    data.use_observation_normalization()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    contract = saved["contract"]
    for key, path in (
        ("observation_manifest", data.root / "SHA256SUMS"),
        ("stats", data.root / "statistics.npz"),
        ("grid", data.root / "grid.npz"),
        ("reference", args.reference),
    ):
        if digest(path) != contract[key]:
            raise ValueError(f"Changed input: {key}")
    model = make_model(data.grid["names"].tolist(), "cuda", steps=args.steps)
    model.load_state_dict(saved["model"], strict=True)
    paths = data.paths(args.split)
    recording = RecordingForecast(model, data, paths, args.output).eval()
    with torch.no_grad():
        metrics, value = evaluate_point_metrics(
            recording, data, paths, json.loads(args.reference.read_text()), members=8
        )
    atomic_json(
        dict(
            step=saved["step"],
            counts=saved["counts"],
            checkpoint_sha256=digest(args.checkpoint),
            training_contract=contract,
            evaluator=os.environ["SAMUDRA_CODE_COMMIT"],
            sampling_steps=args.steps,
            split=args.split,
            score=value,
            metrics=metrics,
            files={p.name: digest(p) for p in args.output.glob("*.npz")},
        ),
        args.output / "COMPLETE.json",
    )


if __name__ == "__main__":
    main()
