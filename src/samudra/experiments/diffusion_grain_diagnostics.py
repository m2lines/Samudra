# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Frozen-latent sampler refinement and paired clean-field denoising tests."""

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from samudra.experiments.diffusion_correlated_pilot import build_wave
from samudra.experiments.diffusion_endpoint_maps import CHANNELS, native_sequence
from samudra.experiments.diffusion_latent_report import load_latent_checkpoint
from samudra.experiments.diffusion_noise import diffusion_noise
from samudra.experiments.diffusion_report import verified_annual_origins
from samudra.experiments.observation_annual import inputs, read_origin
from samudra.experiments.observation_pilot import atomic_json, atomic_torch, digest
from samudra.experiments.observation_training import Samples
from samudra.experiments.surface_state import advance_season
from samudra.rust_data import create_rust_io_runtime, native_om4_source
from samudra.utils.location import LocalLocation

SIGMAS = (0.002, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 80.0)


def diagnose(model, states, references, data, mask, output, clean=None):
    """Hold cached states and initial Gaussian draws fixed across solver choices."""
    output.mkdir(parents=True, exist_ok=True)
    indices = [list(data.grid["names"]).index(c) for c in CHANNELS]
    atomic_torch({k: v.cpu() for k, v in states.items()}, output / "latents.pt")
    common = dict(
        channels=np.array(CHANNELS),
        mask=mask[indices].cpu().numpy(),
        lat=data.grid["lat"],
        lon=data.grid["lon"],
    )
    for lead, state in states.items():
        for steps in (16, 32, 64, 128):
            model.sampling_steps = steps
            generator = torch.Generator(device=data.device).manual_seed(4041729)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                samples = torch.stack(
                    [model.readout(state, mask, generator)[:, -1] for _ in range(8)]
                ).float()
            values = data.physical(samples)[:, 0, indices].cpu().numpy()
            if not np.isfinite(values).all():
                raise FloatingPointError("Nonfinite solver diagnostic")
            np.savez_compressed(
                output / f"day{lead}-steps{steps}.npz",
                members=values,
                reference=references[lead],
                **common,
            )
        if clean is None or lead != 30:
            continue
        target = clean.flatten(1, 2)
        joint_mask = mask.repeat(2, 1, 1)
        errors, noises = [], []
        for sigma in SIGMAS:
            rng = torch.Generator(device=data.device).manual_seed(4041730)
            by_sigma, noise_sigma = [], []
            for _ in range(4):
                noise = diffusion_noise(
                    target.shape,
                    joint_mask,
                    rng,
                    model.decoder.noise_correlation,
                    model.decoder.paired_noise_draws,
                )
                noisy = (target + sigma * noise) * joint_mask
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    predicted = model.decoder(
                        noisy,
                        torch.tensor([sigma], device=data.device),
                        state.flatten(1, 2),
                        joint_mask,
                    )
                error = (predicted.float() - target).reshape_as(clean)[:, -1:]
                # Store normalized errors and injected noise, not physical offsets.
                by_sigma.append(error[0, 0, indices].cpu().numpy())
                noise_sigma.append(
                    noise.reshape_as(clean)[0, -1, indices].cpu().numpy()
                )
            errors.append(by_sigma)
            noises.append(noise_sigma)
        np.savez_compressed(
            output / "denoising.npz",
            errors=np.array(errors),
            noise=np.array(noises),
            sigmas=np.array(SIGMAS),
            scale=data.std[0, 0, indices, 0, 0].cpu().numpy(),
            **common,
        )
    atomic_json(
        dict(
            files={
                p.name: digest(p)
                for p in sorted(output.iterdir())
                if p.suffix in (".npz", ".pt")
            }
        ),
        output / "COMPLETE.json",
    )
    print(f"Completed {output}", flush=True)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observations", action="store_true")
    args = parser.parse_args()
    args.export_only = True
    args.output.mkdir(parents=True, exist_ok=True)
    data = Samples(args.root / "data/observations", "cuda")
    data.use_observation_normalization()
    model, signature = load_latent_checkpoint(args.checkpoint, data)
    for path, key in [
        (args.root / "DATA_READY.json", "data_sha256"),
        (data.root / "SHA256SUMS", "observation_manifest_sha256"),
        (data.root / "statistics.npz", "observation_statistics_sha256"),
    ]:
        if digest(path) != signature[key]:
            raise ValueError("Data contract changed")
    protocol = dict(
        evaluator=os.environ["SAMUDRA_CODE_COMMIT"],
        checkpoint=str(args.checkpoint),
        checkpoint_sha256=digest(args.checkpoint),
        signature=signature,
        sampler_steps=[16, 32, 64, 128],
        members=8,
        denoising_sigmas=SIGMAS,
        denoising_draws=4,
        seed=4041729,
        mask="native wet mask; report global and +/-60 separately",
    )
    atomic_json(protocol, args.output / "protocol.json")
    indices = [list(data.grid["names"]).index(c) for c in CHANNELS]
    wave = build_wave(args, data)
    try:
        source = native_om4_source(
            wave.context_bundle.inference_source,
            LocalLocation(path=Path(wave.args.data_root) / "OM4.zarr"),
            create_rust_io_runtime(4),
        )
        dataset = wave.dataset(source, steps=6)
        wave.prepare(dataset)
        for year in (2015, 2018, 2021):
            index = next(
                i
                for i in range(len(dataset))
                if source.time.values[i + 18].year == year
            )
            surface, past, context, truth, forcing, labels = native_sequence(
                wave, dataset, index
            )
            contexts = torch.stack(
                [advance_season(context, (i + 1) * 5) for i in range(73)], 1
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                initial, _, _ = model.encode_native(surface, past, context, wave.mask)
                states = model.processor.rollout(initial, forcing, contexts)
            reference = {
                lead: data.physical(labels[:, j - 1 : j])[0, 0, indices].cpu().numpy()
                for lead, j in [(30, 6), (365, 73)]
            }
            diagnose(
                model,
                {30: states[6], 365: states[73]},
                reference,
                data,
                wave.mask,
                args.output / f"om4-{year}",
                clean=labels[:, 4:6],
            )
    finally:
        if wave.run:
            wave.run.finish()
    if args.observations:
        for path in verified_annual_origins(
            args.root, dict(staged_data_manifest_sha256=signature["data_sha256"])
        ):
            description, raw = read_origin(path.parent)
            surface, atmosphere, contexts, mask, validity = inputs(data, raw)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                adapted = model.adapt(atmosphere)
                initial, _, _ = model.encode_native(
                    surface, adapted[:, :19], contexts[:, 18], mask, validity
                )
                states = model.processor.rollout(
                    initial, adapted[:, 19:], contexts[:, 19:]
                )
            reference = {}
            for lead, j in [(30, 6), (365, 73)]:
                reference[lead] = np.full(
                    (4, *mask.shape[-2:]), np.nan, dtype=np.float32
                )
                reference[lead][:2] = raw["surface"][19 + j - 1]
            diagnose(
                model,
                {30: states[6], 365: states[73]},
                reference,
                data,
                mask,
                args.output / ("obs-" + description["origin"]),
            )
    atomic_json(protocol, args.output / "COMPLETE.json")


if __name__ == "__main__":
    main()
