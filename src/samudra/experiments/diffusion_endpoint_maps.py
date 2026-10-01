# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Retain individual latent readouts at day 30/365 without decoding unused leads."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from samudra.experiments.diffusion_checkpoint_status import verify_checkpoint
from samudra.experiments.diffusion_latent_report import load_latent_checkpoint
from samudra.experiments.diffusion_report import verified_annual_origins
from samudra.experiments.initializer_wave import InitializerWave
from samudra.experiments.observation_annual import inputs, read_origin
from samudra.experiments.observation_pilot import atomic_json, digest
from samudra.experiments.observation_training import Samples
from samudra.experiments.surface_state import advance_season
from samudra.rust_data import create_rust_io_runtime, native_om4_source
from samudra.utils.location import LocalLocation

CHANNELS = ("thetao_0", "zos", "thetao_9", "so_9")
LEADS = (6, 73)


def selected_readouts(
    model, states, mask, known, anchor, generator, members=8, leads=LEADS
):
    """Preserve the full forecast RNG stream while skipping unused Heun decodes.

    sample_joint draws exactly one full-state Gaussian per readout; decoded fields
    never enter recurrence. This retains identical requested draws and final RNG.
    """
    if not model.stochastic or max(leads) >= len(states):
        raise ValueError("Require stochastic latent trajectory covering every lead")
    result = []
    for _ in range(members):
        selected = []
        for step, state in enumerate(states):
            if step in leads:
                selected.append(model.readout(state, mask, generator)[:, -1].float())
            else:
                torch.randn(
                    (state.shape[0], 2 * model.channels, *mask.shape[-2:]),
                    device=state.device,
                    generator=generator,
                )
        result.append(torch.stack(selected, 1))
    return torch.stack(result)


def export(path, values, reference, data, metadata):
    indices = [list(data.grid["names"]).index(c) for c in CHANNELS]
    physical = data.physical(values)[:, 0, :, indices].cpu().numpy()
    if not np.isfinite(physical).all():
        raise ValueError("Nonfinite endpoint readouts")
    np.savez_compressed(
        path,
        members=physical,
        reference=reference,
        channels=np.array(CHANNELS),
        leads_days=np.array(LEADS) * 5,
        mask=data.grid["mask"][indices],
        lat=data.grid["lat"],
        lon=data.grid["lon"],
    )
    atomic_json(metadata, path.with_suffix(".json"))
    print(json.dumps(dict(event="endpoint_export", file=str(path))), flush=True)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", choices=("observation", "om4"), required=True)
    parser.add_argument("--stopped-run", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = Samples(args.root / "data/observations", "cuda")
    data.use_observation_normalization()
    model, signature = load_latent_checkpoint(args.checkpoint, data)
    status = verify_checkpoint(
        args.checkpoint,
        "PRETRAIN_COMPLETE.json"
        if signature["phase"] == "om4"
        else "OBSERVATION_COMPLETE.json",
        stopped=args.stopped_run,
    )
    for path, key in [
        (args.root / "DATA_READY.json", "data_sha256"),
        (data.root / "SHA256SUMS", "observation_manifest_sha256"),
        (data.root / "statistics.npz", "observation_statistics_sha256"),
    ]:
        if digest(path) != signature[key]:
            raise ValueError("Input data contract changed")
    protocol = dict(
        evaluator_commit=os.environ["SAMUDRA_CODE_COMMIT"],
        checkpoint_sha256=digest(args.checkpoint),
        training_protocol=signature,
        training_status=status,
        input=args.input,
        members=8,
        seed=4041729,
        leads_days=[30, 365],
        channels=CHANNELS,
        scope="Independent physical readouts from a deterministic latent trajectory. Native initialization supplies surface history and forcing, never gold interior.",
    )
    atomic_json(protocol, args.output / "protocol.json")
    indices = [list(data.grid["names"]).index(c) for c in CHANNELS]
    if args.input == "observation":
        for path in verified_annual_origins(
            args.root, dict(staged_data_manifest_sha256=signature["data_sha256"])
        ):
            description, raw = read_origin(path.parent)
            surface, atmosphere, contexts, mask, validity = inputs(data, raw)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                adapted = model.adapt(atmosphere)
                initial, known, anchor = model.encode_native(
                    surface, adapted[:, :19], contexts[:, 18], mask, validity
                )
                states = model.processor.rollout(
                    initial, adapted[:, 19:], contexts[:, 19:]
                )
                values = selected_readouts(
                    model,
                    states,
                    mask,
                    known,
                    anchor,
                    torch.Generator(device=data.device).manual_seed(4041729),
                )
            reference = np.full((2, 4, *mask.shape[-2:]), np.nan, dtype=np.float32)
            reference[:, :2] = raw["surface"][[19 + j - 1 for j in LEADS]]
            # Compare recovered member means to the already-published annual run.
            old = (
                args.checkpoint.parent.parent
                / "report-v1/annual"
                / (description["origin"] + ".npz")
            )
            with np.load(old) as z:
                expected = z["state_at_leads"][
                    [list(z["leads_days"]).index(j * 5) for j in LEADS]
                ][:, indices]
            actual = data.physical(values.mean(0))[0, :, indices].cpu().numpy()
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
            export(
                args.output / (description["origin"] + ".npz"),
                values,
                reference,
                data,
                dict(
                    origin=description["origin"],
                    annual_manifest_sha256=digest(path),
                    annual_reference_sha256=digest(old),
                    recovered_mean_max_absolute_difference=float(
                        np.max(np.abs(actual - expected))
                    ),
                    target_midpoints=[str(raw["midpoint"][19 + j - 1]) for j in LEADS],
                ),
            )
    else:
        loader_args = SimpleNamespace(
            arm="D",
            phase="reconstruction",
            seed=1729,
            readers=4,
            data_root=str(args.root / "data/om4_onedeg_v3"),
            output=str(args.output / "loader"),
            name="latent-endpoints",
            wandb_mode="disabled",
            normalization="instance",
            observation_normalization_root=str(data.root),
            fresh_evolution=True,
            initial_checkpoint=str(args.root / "checkpoints/om4-source/selected.pt"),
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
            dataset = wave.dataset(source, steps=73)
            wave.prepare(dataset)
            for year in (2015, 2018, 2021):
                index = next(
                    i
                    for i in range(len(dataset))
                    if source.time.values[i + 18].year == year
                )
                surface, past, context, truth, forcing, labels = wave.model_sample(
                    dataset, [index]
                )
                if forcing.shape[1] != 73:
                    raise ValueError("Native forecast horizon differs")
                contexts = torch.stack(
                    [advance_season(context, (j + 1) * 5) for j in range(73)], 1
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    initial, known, anchor = model.encode_native(
                        surface, past, context, wave.mask
                    )
                    states = model.processor.rollout(initial, forcing, contexts)
                    values = selected_readouts(
                        model,
                        states,
                        wave.mask,
                        known,
                        anchor,
                        torch.Generator(device=data.device).manual_seed(4041729),
                    )
                reference = (
                    data.physical(labels[:, [j - 1 for j in LEADS]])[0, :, indices]
                    .cpu()
                    .numpy()
                )
                export(
                    args.output / f"{year}-native.npz",
                    values,
                    reference,
                    data,
                    dict(
                        origin=str(source.time.values[index + 18]),
                        index=index,
                        target_midpoints=[
                            str(source.time.values[index + 18 + j]) for j in LEADS
                        ],
                    ),
                )
        finally:
            if wave.run:
                wave.run.finish()
    atomic_json(
        dict(
            protocol=protocol,
            files={p.name: digest(p) for p in sorted(args.output.glob("*.npz"))},
        ),
        args.output / "COMPLETE.json",
    )


if __name__ == "__main__":
    main()
