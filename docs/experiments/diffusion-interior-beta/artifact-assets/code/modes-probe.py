# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen-checkpoint OM4 initialization, persistence and true-interior controls."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from samudra.experiments.diffusion_calibration import (
    ensemble_field_statistics,
    point_field_statistics,
)
from samudra.experiments.diffusion_physical import JointPhysicalForecast
from samudra.experiments.diffusion_structure import field_structure
from samudra.experiments.initializer_wave import InitializerWave
from samudra.experiments.observation_model import ObservationTransfer
from samudra.experiments.observation_pilot import atomic_json, digest
from samudra.experiments.surface_adaptation import state_fingerprint
from samudra.rust_data import create_rust_io_runtime, native_om4_source
from samudra.utils.location import LocalLocation


@torch.no_grad()
def velocity_statistics(members, target, mean, std, mask, weights):
    """Native velocity diagnostics in m/s; callers select channels and lead first.

    Members have shape (member,batch,channel,y,x). The physical zero reference
    is independent of the training normalization. Spatial moments are computed
    for each member before averaging and must not be confused with mean-field
    structure. All groups share exactly the same native wet-area support.
    """
    physical = members.float() * std[:, None, None] + mean[:, None, None]
    truth = target.float() * std[:, None, None] + mean[:, None, None]
    return dict(
        ensemble=ensemble_field_statistics(physical, truth, mask, weights, 1),
        zero_velocity=point_field_statistics(
            torch.zeros_like(truth), truth, mask, weights, 1
        ),
        members_structure=field_structure(physical, mask, weights),
        mean_structure=field_structure(physical.mean(0), mask, weights),
        truth_structure=field_structure(truth, mask, weights),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--velocity-diagnostics", action="store_true")
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
            width=signature.get("decoder_width", 64),
        ).to(wave.device)
        model.load_state_dict(saved["model"], strict=True)
        del saved
        model.eval()
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
            zero_velocity_reference="Zero m/s at each lead, not evolved; emitted for u/v channels only",
        )
        if args.velocity_diagnostics:
            if not model.stochastic:
                raise ValueError("Velocity calibration requires an ensemble")
            protocol["velocity_diagnostics"] = (
                "physical m/s; additive calibration and per-field spatial moments; all 24 origins and four leads"
            )
        contract = args.output / "protocol.json"
        if contract.exists() and json.loads(contract.read_text()) != protocol:
            raise ValueError("Existing control protocol differs")
        atomic_json(protocol, contract)
        c = wave.names.index("so_9")
        selected = [wave.names.index(n) for n in ("so_9", "thetao_9", "uo_9")]
        index = indices[0]
        surface, past, context, truth, forcing, labels = wave.model_sample(
            dataset, [index]
        )
        from samudra.experiments.observation_training import Samples

        obsdata = Samples(args.root / "data/observations", "cuda")
        obsdata.use_observation_normalization()
        obspath = next(p for p in obsdata.paths("test") if p.stem == "2015-01")
        obssample = obsdata.load(obspath)
        np.savez_compressed(
            args.output / "truth.npz",
            truth=(
                truth.float() * wave.std[None, None, :, None, None]
                + wave.mean[None, None, :, None, None]
            )
            .cpu()
            .numpy()[:, :, selected],
            mask=wave.mask[selected].cpu().numpy(),
            lat=wave.lat.cpu().numpy(),
            names=np.array([wave.names[i] for i in selected]),
        )
        rows = []
        model.eval()
        with torch.inference_mode():
            future = model.core.adapt(obssample["atmosphere"][:, 19:])
            ctx = obssample["contexts"][:, 19]

            def evolve(x):
                return model.core.evolution(x, future[:, :1], ctx, wave.mask, 1)

            # Local linear response around real OM4 states; centered differences.
            y = torch.arange(180, device=wave.device)[:, None]
            x = torch.arange(360, device=wave.device)[None, :]
            for wavelength in (4, 8, 16, 32):
                for sign in (-1, 1):
                    mode = (
                        torch.sin(2 * torch.pi * (x + sign * y) / wavelength)
                        * wave.mask[c]
                    )
                    perturb = torch.zeros_like(truth)
                    perturb[:, :, c] = 0.1 * mode
                    response = (evolve(truth + perturb) - evolve(truth - perturb)) / 2
                    patch = response[0, c, 70:100, 180:240]
                    gain = float(
                        patch.square().mean().sqrt()
                        / (0.1 * mode[70:100, 180:240]).square().mean().sqrt()
                    )
                    rows.append(dict(wavelength=wavelength, sign=sign, gain=gain))
            print(json.dumps(rows), flush=True)
            atomic_json(rows, args.output / "mode-gains.json")
    finally:
        if wave.run:
            wave.run.finish()


if __name__ == "__main__":
    main()
