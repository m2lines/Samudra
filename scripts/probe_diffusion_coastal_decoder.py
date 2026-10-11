# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Paired native-OM4 decoder probes; no training and no checkpoint selection.

Crossed decoder/latent tests are interventions, not deployable hybrids: the two
fine-tuned encoders can use different latent coordinates. Nearest-wet land fill
changes only the stem input, preserves the EDM skip and output masks, and is an
inference-only counterfactual, not evidence for a valid trained alternative.
"""

import argparse
import gc
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt  # type: ignore[attr-defined]

from samudra.experiments.diffusion_correlated_pilot import build_wave
from samudra.experiments.diffusion_global import make_model, rng
from samudra.experiments.diffusion_noise import diffusion_noise
from samudra.experiments.observation_pilot import atomic_json, digest
from samudra.experiments.observation_training import Samples
from samudra.experiments.surface_state import advance_season
from samudra.rust_data import create_rust_io_runtime, native_om4_source
from samudra.utils.location import LocalLocation

CHANNELS = [38, 76]
LEADS = [6, 73]


def geometry(mask):
    """Nearest wet indices and distance to land, with periodic longitude only."""
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape[-2:]
    nearest, distances = [], []
    for wet in mask:
        if not wet.any() or wet.all():
            raise ValueError("Probe expects mixed wet/dry channels")
        tiled = np.tile(wet, (1, 3))
        indices = distance_transform_edt(~tiled, return_indices=True)[1]
        nearest.append(
            indices[0, :, width : 2 * width] * width
            + indices[1, :, width : 2 * width] % width
        )
        distances.append(distance_transform_edt(tiled)[:, width : 2 * width])
    return np.stack(nearest), np.stack(distances)


@contextmanager
def fill_stem(decoder, indices, enabled):
    """Extend scaled noisy input across land before convolution, nowhere else."""
    if not enabled:
        yield
        return

    def prehook(module, args):
        value = args[0]
        filled = value.flatten(-2).gather(
            -1, indices.flatten(-2)[None].expand(value.shape[0], -1, -1)
        )
        return (filled.reshape_as(value),)

    handle = decoder.stem.register_forward_pre_hook(prehook)
    try:
        yield
    finally:
        handle.remove()


def metrics(prediction, reference, mask, latitude, distance):
    """Physical-unit errors and field moments; area weighted per coast bin."""
    prediction, reference = np.asarray(prediction), np.asarray(reference)
    if not np.isfinite(prediction).all() or not np.isfinite(reference).all():
        raise FloatingPointError("Nonfinite probe fields")
    weights = np.maximum(np.cos(np.deg2rad(latitude)), 0)[None, :, None] * mask
    bins = {
        "all": mask,
        "0-2": mask & (distance <= 2),
        "2-5": mask & (distance > 2) & (distance <= 5),
        "5-10": mask & (distance > 5) & (distance <= 10),
        "over-10": mask & (distance > 10),
    }
    result = {}
    for label, cells in bins.items():
        channels: list[dict[str, float] | None] = []
        for channel in range(len(CHANNELS)):
            weight = weights[channel] * cells[channel]
            mass = weight.sum()
            if mass <= 0:
                channels.append(None)
                continue
            weight = weight / mass
            field, gold = prediction[channel], reference[channel]
            delta = field - gold
            mean = float((field * weight).sum())
            channels.append(
                dict(
                    cells=int(cells[channel].sum()),
                    rmse=float(np.sqrt((delta**2 * weight).sum())),
                    bias=float((delta * weight).sum()),
                    mean=mean,
                    spatial_std=float(np.sqrt(((field - mean) ** 2 * weight).sum())),
                    min=float(field[cells[channel]].min()),
                    max=float(field[cells[channel]].max()),
                )
            )
        result[label] = dict(zip(("sst", "ssh"), channels, strict=True))
    return result


def bound_latent(state, reference):
    """Inference-only hard counterpart of the trained soft RMS excess penalty."""
    scale = (
        reference.float().square().mean((-2, -1), keepdim=True).sqrt().clamp_min(0.1)
    )
    rms = state.float().square().mean((-2, -1), keepdim=True).sqrt()
    factor = (2 * scale / rms.clamp_min(1e-12)).clamp_max(1)
    bounded = (state.float() * factor).to(state.dtype)
    return bounded, dict(
        fraction_clamped=float((factor < 1).float().mean()),
        pre_rms=float(state.float().square().mean().sqrt()),
        post_rms=float(bounded.float().square().mean().sqrt()),
    )


def load_model(path, data):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    for key, file in (
        ("observation_manifest", "SHA256SUMS"),
        ("stats", "statistics.npz"),
        ("grid", "grid.npz"),
    ):
        if saved["contract"][key] != digest(data.root / file):
            raise ValueError(f"Checkpoint/data mismatch: {key}")
    model = make_model(data.grid["names"].tolist(), "cuda").eval()
    model.load_state_dict(saved["model"], strict=True)
    provenance = dict(
        path=str(path),
        sha256=digest(path),
        counts=saved["counts"],
        intervention=saved.get("intervention"),
        step=saved["step"],
        contract=saved["contract"],
    )
    return model, provenance


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--replay-bound", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--land-fill", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[4041735, 4041736])
    args = parser.parse_args()
    args.control = args.control or args.root / "runs/control/step-0256.pt"
    args.replay_bound = (
        args.replay_bound or args.root / "runs/replay-bound/step-0256.pt"
    )
    if (args.output / "COMPLETE.json").exists():
        raise FileExistsError("Probe is already complete; do not duplicate evaluation")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    data = Samples(
        args.root / "data/observations",
        "cuda",
        surface_fill="zero",
        global_observations=True,
    )
    data.use_observation_normalization()
    models, provenance = {}, {}
    for label, path in (("control", args.control), ("replay-bound", args.replay_bound)):
        models[label], provenance[label] = load_model(path, data)
    if provenance["control"]["counts"] != provenance["replay-bound"]["counts"]:
        raise ValueError("Checkpoint training counts differ")
    wave = build_wave(
        SimpleNamespace(root=args.root, output=args.output, export_only=True), data
    )
    del wave.model
    del wave.initializer
    wave.initializer = SimpleNamespace(surface=models["control"].surface)
    gc.collect()
    source = native_om4_source(
        wave.context_bundle.inference_source,
        LocalLocation(path=args.root / "data/om4_onedeg_v3/OM4.zarr"),
        create_rust_io_runtime(4),
    )
    dataset = wave.dataset(source, steps=73)
    index = next(
        i for i in range(len(dataset)) if source.time.values[i + 18].year == 2015
    )
    surface, past, context, truth, forcing, targets = wave.model_sample(
        dataset, [index]
    )
    latent_states, latent_norms = {}, {}
    bounded_states = {}
    bound_records: dict[str, Any] = {}
    joint_mask = wave.mask.repeat(2, 1, 1)
    nearest, distance = geometry(wave.mask.cpu().numpy())
    nearest = torch.as_tensor(
        np.tile(nearest, (2, 1, 1)), device="cuda", dtype=torch.long
    )
    wet = wave.mask.cpu().numpy()[CHANNELS].astype(bool)
    distance = distance[CHANNELS]
    lat = data.grid["lat"]
    mean = wave.mean[CHANNELS, None, None]
    std = wave.std[CHANNELS, None, None]

    def physical(pair):
        return (
            (
                pair.reshape(1, 2, 77, *wet.shape[-2:])[0, -1, CHANNELS].float() * std
                + mean
            )
            .cpu()
            .numpy()
        )

    with torch.autocast("cuda", dtype=torch.bfloat16):
        for label, model in models.items():
            state, _, _ = model.encode_native(surface, past, context, wave.mask)
            states, norms = (
                {0: state.clone()},
                [float(state.float().square().mean().sqrt())],
            )
            for step in range(73):
                state = model.processor(
                    state,
                    forcing[:, step],
                    advance_season(context, (step + 1) * 5),
                    task="om4",
                )
                norms.append(float(state.float().square().mean().sqrt()))
                if step + 1 in LEADS:
                    states[step + 1] = state.clone()
            latent_states[label], latent_norms[label] = states, norms
            recurrent, recurrent_records = {}, []
            state = states[0].clone()
            for step in range(73):
                proposal = model.processor(
                    state,
                    forcing[:, step],
                    advance_season(context, (step + 1) * 5),
                    task="om4",
                )
                state, diagnostics = bound_latent(proposal, states[0])
                recurrent_records.append(dict(lead_days=(step + 1) * 5, **diagnostics))
                if step + 1 in LEADS:
                    recurrent[step + 1] = state.clone()
            bounded_states[label] = recurrent
            bound_records[label] = dict(recurrent=recurrent_records, readout={})

        maps, records = {}, []
        for lead in LEADS:
            clean = torch.stack(
                (targets[:, lead - 2], targets[:, lead - 1]), 1
            ).flatten(1, 2)
            reference = physical(clean)
            maps[f"reference_day{lead * 5}"] = reference
            noise = diffusion_noise(
                clean.shape, joint_mask, rng("cuda", 991000 + lead), 0.5
            )
            for label, model in models.items():
                for conditioning in ("correct", "zero", "origin"):
                    state = latent_states[label][lead]
                    if conditioning == "zero":
                        state = torch.zeros_like(state)
                    elif conditioning == "origin":
                        state = latent_states[label][0]
                    for sigma in (0.05, 0.3, 1.0, 5.0):
                        noisy = clean + sigma * noise
                        decoded = model.decoder(
                            noisy,
                            noisy.new_full((1,), sigma),
                            state.flatten(1, 2),
                            joint_mask,
                        )
                        key = (
                            f"denoise_{label}_{conditioning}_day{lead * 5}_sigma{sigma}"
                        )
                        maps[key] = physical(decoded)
                        records.append(
                            dict(
                                key=key,
                                kind="denoise",
                                decoder=label,
                                conditioning=conditioning,
                                lead_days=lead * 5,
                                sigma=sigma,
                                input_metrics=metrics(
                                    physical(noisy), reference, wet, lat, distance
                                ),
                                metrics=metrics(
                                    maps[key], reference, wet, lat, distance
                                ),
                            )
                        )
            for decoder_label, model in models.items():
                for latent_label, states in latent_states.items():
                    for seed in args.seeds:
                        decoded = model.readout(
                            states[lead], wave.mask, rng("cuda", seed)
                        )
                        key = f"sample_decoder-{decoder_label}_latent-{latent_label}_day{lead * 5}_seed{seed}"
                        maps[key] = physical(decoded)
                        records.append(
                            dict(
                                key=key,
                                kind="sample",
                                decoder=decoder_label,
                                conditioning=latent_label,
                                lead_days=lead * 5,
                                seed=seed,
                                metrics=metrics(
                                    maps[key], reference, wet, lat, distance
                                ),
                            )
                        )
                        print(
                            json.dumps(
                                dict(
                                    event="sample_complete",
                                    key=key,
                                    elapsed_seconds=time.monotonic() - started,
                                )
                            ),
                            flush=True,
                        )
                readout_bound, bound_diagnostics = bound_latent(
                    latent_states[decoder_label][lead], latent_states[decoder_label][0]
                )
                bound_records[decoder_label]["readout"][str(lead * 5)] = (
                    bound_diagnostics
                )
                for variant, probe_state in (
                    ("sample_readout_rms_bound", readout_bound),
                    ("sample_recurrent_rms_bound", bounded_states[decoder_label][lead]),
                ):
                    for seed in args.seeds:
                        decoded = model.readout(
                            probe_state, wave.mask, rng("cuda", seed)
                        )
                        key = f"{variant}_{decoder_label}_day{lead * 5}_seed{seed}"
                        maps[key] = physical(decoded)
                        records.append(
                            dict(
                                key=key,
                                kind=variant,
                                decoder=decoder_label,
                                conditioning=decoder_label,
                                lead_days=lead * 5,
                                seed=seed,
                                metrics=metrics(
                                    maps[key], reference, wet, lat, distance
                                ),
                            )
                        )
                        print(
                            json.dumps(
                                dict(
                                    event="sample_complete",
                                    key=key,
                                    elapsed_seconds=time.monotonic() - started,
                                )
                            ),
                            flush=True,
                        )
                if args.land_fill and lead == 73:
                    with fill_stem(model.decoder, nearest, True):
                        for seed in args.seeds:
                            decoded = model.readout(
                                latent_states[decoder_label][lead],
                                wave.mask,
                                rng("cuda", seed),
                            )
                            key = f"sample_landfill_{decoder_label}_day{lead * 5}_seed{seed}"
                            maps[key] = physical(decoded)
                            records.append(
                                dict(
                                    key=key,
                                    kind="sample_landfill",
                                    decoder=decoder_label,
                                    conditioning=decoder_label,
                                    lead_days=lead * 5,
                                    seed=seed,
                                    metrics=metrics(
                                        maps[key], reference, wet, lat, distance
                                    ),
                                )
                            )
            atomic_json(records, args.output / "metrics.json")
            np.savez_compressed(
                args.output / "maps.npz",
                **maps,
                mask=wet,
                distance_to_land=distance,
                lat=lat,
                lon=data.grid["lon"],
                channels=np.array(["sst", "ssh"]),
            )
    latent_arrays = {
        f"{label}_day{lead * 5}": value.float().cpu().numpy()
        for label, states in latent_states.items()
        for lead, value in states.items()
    }
    latent_arrays.update(
        {
            f"recurrent_bound_{label}_day{lead * 5}": value.float().cpu().numpy()
            for label, states in bounded_states.items()
            for lead, value in states.items()
        }
    )
    np.savez_compressed(args.output / "latents.npz", **latent_arrays)
    protocol = dict(
        checkpoints=provenance,
        origin=str(source.time.values[index + 18]),
        producer=os.environ.get("SAMUDRA_CODE_COMMIT", "unspecified"),
        script_sha256=digest(Path(__file__)),
        leads_days=[30, 365],
        seeds=args.seeds,
        sampling_steps=32,
        noise_correlation=0.5,
        denoising_sigmas=[0.05, 0.3, 1.0, 5.0],
        denoising_noise_seeds=[991006, 991073],
        latent_rms=latent_norms,
        latent_bound=bound_records,
        latent_bound_definition="Per batch/slot/channel spatial RMS hard capped at 2 times encoded-origin RMS clamped to minimum 0.1. Applied either only before decoding or after each processor step; never in place.",
        channels=["thetao_0", "zos"],
        coast_distance="Euclidean grid cells to dry cells, periodic longitude only; not physical distance",
        target_pair="At lead L, targets[L/5-2] and targets[L/5-1]: physical days L-5 and L; score newest frame only",
        land_fill="Inference-only nearest-wet extension of scaled noisy stem input; EDM skip, output masking and sampler unchanged"
        if args.land_fill
        else None,
        caveats=[
            "One OM4 origin, two seeds by default; mechanistic diagnostic, not a skill benchmark",
            "Decoder crossing can fail from latent-coordinate mismatch",
            "Zero and origin latents are counterfactual conditioning, not valid forecasts",
            "Land-fill evaluation alone cannot validate retraining with changed boundaries",
            "Hard RMS bounding is an inference/training mismatch, not GroupNorm and not evidence for a retrained normalization architecture",
        ],
        elapsed_seconds=time.monotonic() - started,
        files={
            p.name: digest(p)
            for p in (
                args.output / "metrics.json",
                args.output / "maps.npz",
                args.output / "latents.npz",
            )
        },
    )
    atomic_json(protocol, args.output / "COMPLETE.json")
    print(
        json.dumps(dict(event="complete", elapsed_seconds=protocol["elapsed_seconds"])),
        flush=True,
    )


if __name__ == "__main__":
    main()
