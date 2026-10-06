# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Final global annual point metrics and individual conditional readout exports."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from samudra.experiments.diffusion_global import make_model
from samudra.experiments.observation_annual import evaluate_origins
from samudra.experiments.observation_pilot import atomic_json, digest
from samudra.experiments.observation_training import Samples

CHANNELS = [38, 76, 47, 66]
ORIGINS = ["2015-01-01", "2018-01-01", "2021-01-01"]


class AnnualMembers(nn.Module):
    """Export member fields before averaging; no coherent trajectory claim."""

    def __init__(self, model, data, origins, output, members=8, seed=4041729):
        super().__init__()
        self.model, self.data = model, data
        self.origins, self.output = origins, output
        self.members, self.seed, self.completed = members, seed, 0

    def forecast(self, *args):
        surface = args[0]
        trajectories, initial = self.model.forecast(
            *args,
            members=self.members,
            generator=torch.Generator(device=surface.device).manual_seed(self.seed),
        )
        if trajectories.shape[:4] != (self.members, 1, 73, 77):
            raise ValueError("Unexpected annual ensemble shape")
        # Select four channels before converting to physical units, avoiding a
        # second full 8 x 73 x 77 float32 ensemble allocation.
        mean = self.data.grid["mean"][CHANNELS][None, None, :, None, None]
        std = self.data.grid["std"][CHANNELS][None, None, :, None, None]
        fields = trajectories[:, 0, :, CHANNELS].float().cpu().numpy() * std + mean
        initial_fields = initial[:, 0, :, CHANNELS].float().cpu().numpy() * std + mean
        if not np.isfinite(fields).all() or not np.isfinite(initial_fields).all():
            raise ValueError("Nonfinite annual member output")
        origin = self.origins[self.completed].parent.name
        np.savez_compressed(
            self.output / f"members-{origin}.npz",
            fields=fields,
            initial=initial_fields,
            leads_days=np.arange(1, 74) * 5,
            channel_names=self.data.grid["names"][CHANNELS],
            lat=self.data.grid["lat"],
            lon=self.data.grid["lon"],
            mask=self.data.grid["mask"][CHANNELS],
        )
        self.completed += 1
        return trajectories.mean(0, dtype=torch.float32), initial.mean(
            0, dtype=torch.float32
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--annual-data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=32)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = Samples(args.data, "cuda", surface_fill="zero", global_observations=True)
    data.use_observation_normalization()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if saved["step"] != 16000 or saved["counts"] != {
        "om4": 8000,
        "observation": 8000,
    }:
        raise ValueError("This held-out annual report requires final 8k/8k weights")
    contract = saved["contract"]
    for key, path in (
        ("observation_manifest", data.root / "SHA256SUMS"),
        ("stats", data.root / "statistics.npz"),
        ("grid", data.root / "grid.npz"),
        ("reference", args.reference),
    ):
        if digest(path) != contract[key]:
            raise ValueError(f"Changed input: {key}")
    origins = sorted((args.annual_data / "test").glob("*/COMPLETE.json"))
    if [p.parent.name for p in origins] != ORIGINS:
        raise ValueError("Incomplete frozen annual test cohort")
    # read_origin/evaluate_origins verify every payload hash and exact timestamp.
    protocol = dict(
        checkpoint_sha256=digest(args.checkpoint),
        training_contract=contract,
        counts=saved["counts"],
        step=saved["step"],
        evaluator=os.environ["SAMUDRA_CODE_COMMIT"],
        sampling_steps=args.steps,
        members=8,
        seed=4041729,
        annual_manifests={p.parent.name: digest(p) for p in origins},
        domain="All finite wet latitudes; zero-normalized missing surface inputs",
        uncertainty="Independent conditional diffusion readouts along one deterministic latent path; not coherent ensemble trajectories",
        member_fields="SST, SSH, temperature at550m, salinity at550m at all73 five-day leads; member means are separate point outputs",
    )
    manifest = args.output / "input.json"
    if manifest.exists() and json.loads(manifest.read_text()) != protocol:
        raise ValueError("Existing annual output has a different protocol")
    atomic_json(protocol, manifest)
    model = make_model(data.grid["names"].tolist(), "cuda", steps=args.steps)
    model.load_state_dict(saved["model"], strict=True)
    recording = AnnualMembers(model, data, origins, args.output).eval()
    evaluate_origins(recording, data, origins, args.output, protocol)
    atomic_json(
        dict(
            inputs=protocol,
            files={p.name: digest(p) for p in args.output.glob("*.npz")},
        ),
        args.output / "MEMBERS_COMPLETE.json",
    )


if __name__ == "__main__":
    main()
