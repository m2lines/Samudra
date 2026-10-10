# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Sparse-readout full-year OM4 probes, including the untouched parent."""

import argparse
import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from samudra.experiments.diffusion_correlated_pilot import build_wave
from samudra.experiments.diffusion_global import make_model, rng
from samudra.experiments.observation_pilot import atomic_json, digest
from samudra.experiments.observation_training import Samples
from samudra.experiments.surface_state import advance_season, channel_mse
from samudra.rust_data import create_rust_io_runtime, native_om4_source
from samudra.utils.location import LocalLocation


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = Samples(
        args.root / "data/observations",
        "cuda",
        surface_fill="zero",
        global_observations=True,
    )
    data.use_observation_normalization()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    for key, file in (
        ("observation_manifest", "SHA256SUMS"),
        ("stats", "statistics.npz"),
        ("grid", "grid.npz"),
    ):
        if saved["contract"][key] != digest(data.root / file):
            raise ValueError(f"Changed data: {key}")
    model = make_model(data.grid["names"].tolist(), "cuda").eval()
    model.load_state_dict(saved["model"], strict=True)
    protocol = dict(
        checkpoint_sha256=digest(args.checkpoint),
        counts=saved["counts"],
        intervention=saved.get("intervention"),
        producer=os.environ["SAMUDRA_CODE_COMMIT"],
        members=8,
        sampling_steps=32,
        seed=4041729,
        leads_days=[0, 30, 90, 180, 365],
    )
    del saved
    wave = build_wave(
        SimpleNamespace(root=args.root, output=args.output, export_only=True), data
    )
    del wave.model
    del wave.initializer
    wave.initializer = SimpleNamespace(surface=model.surface)
    gc.collect()
    source = native_om4_source(
        wave.context_bundle.inference_source,
        LocalLocation(path=args.root / "data/om4_onedeg_v3/OM4.zarr"),
        create_rust_io_runtime(4),
    )
    dataset = wave.dataset(source, steps=73)
    channels = [38, 76, 47, 66]
    for year in (2015, 2018, 2021):
        index = next(
            i for i in range(len(dataset)) if source.time.values[i + 18].year == year
        )
        surface, past, context, truth, forcing, targets = wave.model_sample(
            dataset, [index]
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            latent, known, anchor = model.encode_native(
                surface, past, context, wave.mask
            )
            states = [latent]
            for step in range(73):
                states.append(
                    model.processor(
                        states[-1],
                        forcing[:, step],
                        advance_season(context, (step + 1) * 5),
                        task="om4",
                    )
                )
            records, fields, references = {}, [], []
            for lead in (0, 6, 18, 36, 73):
                generator = rng("cuda", 4041729 + lead)
                members = torch.stack(
                    [
                        model.readout(
                            states[lead],
                            wave.mask,
                            generator,
                            known if lead == 0 else None,
                            anchor if lead == 0 else None,
                        )[:, -1]
                        for _ in range(8)
                    ]
                ).float()
                target = truth[:, -1] if lead == 0 else targets[:, lead - 1]
                error = channel_mse(members.mean(0), target, wave.weights)
                member_error = channel_mse(members, target, wave.weights).mean(0)
                spread = channel_mse(members, members.mean(0), wave.weights).mean(0)
                records[str(lead * 5)] = dict(
                    normalized_mse=error.cpu().tolist(),
                    member_normalized_mse=member_error.cpu().tolist(),
                    normalized_spread_variance=spread.cpu().tolist(),
                )
                fields.append(
                    (
                        members[:, 0, channels] * wave.std[channels, None, None]
                        + wave.mean[channels, None, None]
                    )
                    .cpu()
                    .numpy()
                )
                references.append(
                    (
                        target[0, channels] * wave.std[channels, None, None]
                        + wave.mean[channels, None, None]
                    )
                    .cpu()
                    .numpy()
                )
            norms = [float(z.float().square().mean().sqrt()) for z in states]
        if not np.isfinite(np.stack(fields)).all():
            raise FloatingPointError("Nonfinite native probe")
        np.savez_compressed(
            args.output / f"fields-{year}.npz",
            members=np.stack(fields, axis=1),
            reference=np.stack(references),
            channels=data.grid["names"][channels],
            mask=data.grid["mask"][channels],
            lat=data.grid["lat"],
            lon=data.grid["lon"],
            leads_days=np.array([0, 30, 90, 180, 365]),
        )
        atomic_json(
            dict(
                origin=str(source.time.values[index + 18]),
                metrics=records,
                latent_rms=norms,
            ),
            args.output / f"{year}.json",
        )
        print(json.dumps(dict(event="native_probe", year=year)), flush=True)
    atomic_json(
        dict(**protocol, files={p.name: digest(p) for p in args.output.glob("*.npz")}),
        args.output / "COMPLETE.json",
    )


if __name__ == "__main__":
    main()
