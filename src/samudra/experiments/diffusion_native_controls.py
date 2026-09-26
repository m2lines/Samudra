# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen-checkpoint OM4 initialization, persistence and true-interior controls."""

import argparse
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from samudra.experiments.diffusion_physical import JointPhysicalForecast
from samudra.experiments.initializer_wave import InitializerWave, Pair
from samudra.experiments.observation_model import ObservationTransfer
from samudra.experiments.observation_pilot import atomic_json, digest
from samudra.experiments.surface_adaptation import state_fingerprint
from samudra.experiments.surface_state import channel_mse
from samudra.rust_data import create_rust_io_runtime, native_om4_source
from samudra.utils.location import LocalLocation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    signature = saved["protocol"]["signature"]
    phase = signature["phase"]
    marker = "PRETRAIN_COMPLETE.json" if phase == "om4" else "OBSERVATION_COMPLETE.json"
    complete = json.loads((args.checkpoint.parent / marker).read_text())
    if phase not in ("om4", "observation") or signature["arm"] not in ("A", "B"):
        raise ValueError("Require a completed A/B training stage")
    if (
        not complete["state"]["complete"]
        or digest(args.checkpoint) != complete["best_checkpoint_sha256"]
    ):
        raise ValueError("Selected checkpoint differs from completed stage")
    if (
        digest(args.root / "DATA_READY.json")
        != signature["staged_data_manifest_sha256"]
    ):
        raise ValueError("Campaign data audit changed")
    audit = json.loads((args.root / "DATA_READY.json").read_text())
    if (
        digest(args.root / "data/observations/SHA256SUMS")
        != signature["observation_manifest_sha256"]
        or digest(args.root / "data/observations/statistics.npz")
        != audit["observations"]["statistics_sha256"]
    ):
        raise ValueError("Observation normalization contract changed")
    source_checkpoint = args.root / "checkpoints/om4-source/selected.pt"
    if digest(source_checkpoint) != signature["source_sha256"]:
        raise ValueError("Pre-observation source changed")
    args.output.mkdir(parents=True, exist_ok=True)
    loader_args = SimpleNamespace(
        arm="D",
        phase="reconstruction",
        seed=1729,
        readers=4,
        data_root=str(args.root / "data/om4_onedeg_v3"),
        output=str(args.output / "loader"),
        name="native-controls",
        wandb_mode="disabled",
        normalization="instance",
        observation_normalization_root=str(args.root / "data/observations"),
        fresh_evolution=True,
        initial_checkpoint=str(source_checkpoint),
        wave1_root="",
        val_origins=12,
        device_cache=False,
        device_cache_reserve_gib=64,
        batch_size=1,
    )
    wave = InitializerWave(loader_args)
    try:
        core = ObservationTransfer(wave.names, "instance")
        core.initializer, core.evolution = wave.initializer, wave.model.evolution
        model = JointPhysicalForecast(
            core,
            stochastic=signature["arm"] == "B",
            sampling_steps=signature["sampling_steps"],
        ).to(wave.device)
        model.load_state_dict(saved["model"], strict=True)
        del saved
        model.eval()
        native_pair = Pair(torch.nn.Identity(), core.evolution).eval()
        if state_fingerprint(core.evolution) != wave.original_dynamics:
            raise ValueError("Dynamics differ from the frozen pre-observation source")
        source = native_om4_source(
            wave.context_bundle.inference_source,
            LocalLocation(path=Path(loader_args.data_root) / "OM4.zarr"),
            create_rust_io_runtime(4),
        )
        dataset = wave.dataset(source)
        wave.prepare(dataset)
        eligible = [
            i
            for i in range(len(dataset))
            if source.time.values[i + 18].year >= 2015
            and source.time.values[i + 24].year <= 2022
        ]
        indices = [
            eligible[i] for i in np.linspace(0, len(eligible) - 1, 24, dtype=int)
        ]
        if len(set(indices)) != 24:
            raise ValueError("Need 24 distinct held-out origins")
        protocol = dict(
            producer=os.environ["SAMUDRA_CODE_COMMIT"],
            checkpoint_sha256=digest(args.checkpoint),
            training_protocol=signature,
            members=8 if model.stochastic else 1,
            seed=4041729,
            origins=[str(source.time.values[i + 18]) for i in indices],
            scope="Model-world held-out control; native OM4 fluxes; no ERA5 adapter; not observed velocity skill",
            context="Original native Pair.evolve season/forcing convention",
            dynamics_fingerprint=wave.original_dynamics,
        )
        contract = args.output / "protocol.json"
        if contract.exists() and json.loads(contract.read_text()) != protocol:
            raise ValueError("Existing control protocol differs")
        atomic_json(protocol, contract)
        destination = args.output / "metrics.csv"
        temporary = destination.with_suffix(".tmp")
        with temporary.open("w", newline="") as stream, torch.inference_mode():
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "origin",
                    "region",
                    "wet_area",
                    "mode",
                    "lead_days",
                    "channel",
                    "normalized_mse",
                    "physical_mse",
                ],
            )
            writer.writeheader()
            for index, origin in zip(indices, protocol["origins"], strict=True):
                surface, past, context, truth, forcing, labels = wave.model_sample(
                    dataset, [index]
                )
                generator = torch.Generator(device=wave.device).manual_seed(
                    protocol["seed"]
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    latent, known, known_mask = model.encode_native(
                        surface, past, context, wave.mask
                    )
                    initials, trajectories = [], []
                    for _ in range(protocol["members"]):
                        initial = model.initialize(
                            latent, known, known_mask, wave.mask, generator
                        )
                        rollout = native_pair.evolve(
                            initial, forcing, context, wave.mask
                        )
                        initials.append(initial)
                        trajectories.append(torch.cat((initial[:, -1:], rollout), 1))
                    golden = native_pair.evolve(truth, forcing, context, wave.mask)
                members = torch.stack(trajectories).float()
                initial_mean = torch.stack(initials).float().mean(0)
                estimates = dict(
                    inferred_mean=members.mean(0),
                    inferred_member_1=members[0],
                    true_interior=torch.cat((truth[:, -1:], golden), 1),
                    inferred_persistence=initial_mean[:, -1:].expand(-1, 7, -1, -1, -1),
                    true_persistence=truth[:, -1:].expand(-1, 7, -1, -1, -1),
                )
                target = torch.cat((truth[:, -1:], labels), 1)
                regions = {
                    "global": torch.ones_like(wave.lat),
                    "scored_latitudes": (wave.lat.abs() <= 60),
                    "outside_scored_latitudes": (wave.lat.abs() > 60),
                }
                for mode, prediction in estimates.items():
                    if not bool(torch.isfinite(prediction).all()):
                        raise FloatingPointError("Nonfinite native control rollout")
                    for region, selector in regions.items():
                        weights = wave.weights * selector[None, :, None]
                        support = weights.sum((-2, -1)).cpu().numpy()
                        errors = (
                            channel_mse(prediction, target, weights).cpu().numpy()[0]
                        )
                        scales = wave.std.square().cpu().numpy()
                        for lead in (0, 1, 3, 6):
                            for channel, name in enumerate(wave.names):
                                if support[channel] > 0:
                                    writer.writerow(
                                        dict(
                                            origin=origin,
                                            region=region,
                                            wet_area=float(support[channel]),
                                            mode=mode,
                                            lead_days=5 * lead,
                                            channel=name,
                                            normalized_mse=float(errors[lead, channel]),
                                            physical_mse=float(
                                                errors[lead, channel] * scales[channel]
                                            ),
                                        )
                                    )
                print(
                    json.dumps(dict(event="native_control_origin", origin=origin)),
                    flush=True,
                )
        temporary.replace(destination)
        atomic_json(
            dict(**protocol, metrics_sha256=digest(destination)),
            args.output / "COMPLETE.json",
        )
    finally:
        if wave.run:
            wave.run.finish()


if __name__ == "__main__":
    main()
