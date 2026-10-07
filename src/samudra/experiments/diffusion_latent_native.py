# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Native OM4 controls and temporal/member structure for latent diffusion."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from samudra.experiments.diffusion_checkpoint_status import verify_checkpoint
from samudra.experiments.diffusion_global import make_model
from samudra.experiments.diffusion_latent_report import load_latent_checkpoint
from samudra.experiments.diffusion_native_controls import velocity_statistics
from samudra.experiments.initializer_wave import InitializerWave
from samudra.experiments.observation_pilot import atomic_json, digest
from samudra.experiments.observation_training import Samples
from samudra.experiments.surface_state import advance_season, channel_mse
from samudra.rust_data import create_rust_io_runtime, native_om4_source
from samudra.utils.location import LocalLocation


def portable(value):
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
        result = array.astype(object)
        result[~np.isfinite(array)] = None
        return result.tolist()
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    return value


def temporal_statistics(members, truth, weights):
    """Physical increments and adjacent-lead covariance of member deviations.

    Shape is ensemble,batch,time,channel,y,x. These finite-ensemble diagnostics
    are descriptive, not population calibration estimates or an ocean spectrum.
    """

    def reduced(value):
        return (value * weights).sum((-2, -1)) / weights.sum((-2, -1)).clamp_min(1e-12)

    delta = members[:, :, 1:] - members[:, :, :-1]
    truth_delta = truth[:, 1:] - truth[:, :-1]
    anomaly = members - members.mean(0, keepdim=True)
    first, second = anomaly[:, :, :-1], anomaly[:, :, 1:]
    cross = reduced(first * second).mean((0, 1, 2))
    power1 = reduced(first.square()).mean((0, 1, 2))
    power2 = reduced(second.square()).mean((0, 1, 2))
    return dict(
        member_increment_mse=reduced(delta.square()).mean((0, 1, 2)),
        mean_increment_mse=reduced(
            (members.mean(0)[:, 1:] - members.mean(0)[:, :-1]).square()
        ).mean((0, 1)),
        truth_increment_mse=reduced(truth_delta.square()).mean((0, 1)),
        adjacent_member_correlation=cross / (power1 * power2).sqrt().clamp_min(1e-12),
    )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stopped-run", action="store_true")
    parser.add_argument("--global-final", action="store_true")
    args = parser.parse_args()
    data = Samples(
        args.root / "data/observations", "cuda", global_observations=args.global_final
    )
    data.use_observation_normalization()
    if args.global_final:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if saved["step"] != 16000 or saved["counts"] != {
            "om4": 8000,
            "observation": 8000,
        }:
            raise ValueError("Global native report requires final 8k/8k weights")
        signature = saved["contract"]
        for key, path in (
            ("observation_manifest", data.root / "SHA256SUMS"),
            ("stats", data.root / "statistics.npz"),
            ("grid", data.root / "grid.npz"),
        ):
            if digest(path) != signature[key]:
                raise ValueError(f"Changed global native input: {key}")
        model = make_model(data.grid["names"].tolist(), "cuda", steps=32)
        model.load_state_dict(saved["model"], strict=True)
        model.eval()
        training_status = {"step": saved["step"], "counts": saved["counts"]}
    else:
        model, signature = load_latent_checkpoint(args.checkpoint, data)
        marker = (
            "PRETRAIN_COMPLETE.json"
            if signature["phase"] == "om4"
            else "OBSERVATION_COMPLETE.json"
        )
        training_status = verify_checkpoint(
            args.checkpoint, marker, stopped=args.stopped_run
        )
        if (
            digest(args.root / "DATA_READY.json") != signature["data_sha256"]
            or digest(data.root / "SHA256SUMS")
            != signature["observation_manifest_sha256"]
        ):
            raise ValueError("Data contract changed")
        if (
            digest(data.root / "statistics.npz")
            != signature["observation_statistics_sha256"]
        ):
            raise ValueError("Observation scales changed")
    args.output.mkdir(parents=True, exist_ok=True)
    loader_args = SimpleNamespace(
        arm="D",
        phase="reconstruction",
        seed=1729,
        readers=4,
        data_root=str(args.root / "data/om4_onedeg_v3"),
        output=str(args.output / "loader"),
        name="latent-native",
        wandb_mode="disabled",
        normalization="instance",
        observation_normalization_root=str(data.root),
        fresh_evolution=True,
        initial_checkpoint=""
        if args.global_final
        else str(args.root / "checkpoints/om4-source/selected.pt"),
        wave1_root="",
        val_origins=12,
        device_cache=False,
        device_cache_reserve_gib=64,
        batch_size=1,
    )
    wave = InitializerWave(loader_args)
    try:
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
            raise ValueError("Need 24 distinct origins")
        protocol = dict(
            evaluator_commit=os.environ["SAMUDRA_CODE_COMMIT"],
            training_protocol=signature,
            training_status=training_status,
            checkpoint_sha256=digest(args.checkpoint),
            origins=[str(source.time.values[i + 18]) for i in indices],
            members=8,
            seed=4041729,
            leads=[0, 5, 10, 15, 20, 25, 30],
            channels=wave.names,
            global_domain=args.global_final,
            native_store=str(Path(loader_args.data_root) / "OM4.zarr"),
            scope="Model-world OM4 controls, not observed interior/velocity skill. Independent readouts on a shared latent trajectory.",
        )
        contract = args.output / "protocol.json"
        if contract.exists() and json.loads(contract.read_text()) != protocol:
            raise ValueError("Native protocol changed")
        atomic_json(protocol, contract)
        hashes = {}
        for index in indices:
            surface, past, context, truth, forcing, labels = wave.model_sample(
                dataset, [index]
            )
            contexts = torch.stack(
                [advance_season(context, (j + 1) * 5) for j in range(forcing.shape[1])],
                1,
            )
            generator = torch.Generator(device=wave.device).manual_seed(4041729)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                initial, known, anchor = model.encode_native(
                    surface, past, context, wave.mask
                )
                latent_states = model.processor.rollout(initial, forcing, contexts)
                records = []
                for member in range(8):
                    decoded = [
                        model.readout(
                            latent_states[0], wave.mask, generator, known, anchor
                        )[:, -1]
                    ]
                    decoded.extend(
                        model.readout(z, wave.mask, generator)[:, -1]
                        for z in latent_states[1:]
                    )
                    records.append(torch.stack(decoded, 1))
                members = torch.stack(records).float()
            target = torch.cat((truth[:, -1:], labels), 1)
            region = (
                torch.ones_like(wave.lat, dtype=torch.bool)
                if args.global_final
                else wave.lat.abs() <= 60
            )[:, None]
            mask = wave.mask * region
            weights = wave.weights * region
            mse = channel_mse(members.mean(0), target, weights)
            persistence = target[:, :1].expand_as(target)
            persistence_mse = channel_mse(persistence, target, weights)
            velocities = {}
            for lead in (0, 1, 3, 6):
                velocities[str(lead * 5)] = velocity_statistics(
                    members[:, :, lead, :38],
                    target[:, lead, :38],
                    wave.mean[:38],
                    wave.std[:38],
                    mask[:38],
                    weights[:38],
                )
            physical = (
                members * wave.std[None, None, None, :, None, None]
                + wave.mean[None, None, None, :, None, None]
            )
            physical_truth = (
                target * wave.std[None, None, :, None, None]
                + wave.mean[None, None, :, None, None]
            )
            result = dict(
                origin=str(source.time.values[index + 18]),
                normalized_mse=mse,
                physical_mse=mse * wave.std.square(),
                persistence_normalized_mse=persistence_mse,
                velocity=velocities,
                temporal=temporal_statistics(physical, physical_truth, weights),
            )
            destination = args.output / f"origin-{index}.json"
            atomic_json(portable(result), destination)
            hashes[destination.name] = digest(destination)
            if index in (indices[0], indices[12], indices[-1]):
                channels = [wave.names.index(n) for n in ("so_9", "thetao_9", "uo_9")]
                arrays_path = args.output / f"fields-{index}.npz"
                np.savez_compressed(
                    arrays_path,
                    members=physical[:, :, :, channels].cpu().numpy(),
                    truth=physical_truth[:, :, channels].cpu().numpy(),
                    mask=mask[channels].cpu().numpy(),
                    channels=np.array([wave.names[c] for c in channels]),
                )
                hashes[arrays_path.name] = digest(arrays_path)
            print(
                json.dumps(dict(event="latent_native_origin", index=index)), flush=True
            )
        atomic_json(dict(**protocol, files=hashes), args.output / "COMPLETE.json")
    finally:
        if wave.run:
            wave.run.finish()


if __name__ == "__main__":
    main()
