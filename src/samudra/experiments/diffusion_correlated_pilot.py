# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Matched short OM4 training for spatially correlated diffusion noise."""

import argparse
import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from samudra.experiments.diffusion_fit import batch_at, bounded_fit
from samudra.experiments.diffusion_latent_report import load_latent_checkpoint
from samudra.experiments.diffusion_latent_train import native_loss
from samudra.experiments.initializer_wave import InitializerWave
from samudra.experiments.observation_pilot import atomic_json, atomic_torch, digest
from samudra.experiments.observation_training import Samples
from samudra.experiments.surface_state import advance_season
from samudra.rust_data import create_rust_io_runtime, native_om4_source
from samudra.utils.location import LocalLocation


@torch.inference_mode()
def export_members(model, wave, dataset, data, output):
    output.mkdir(parents=True, exist_ok=True)
    model.eval()
    names = ("thetao_0", "zos", "thetao_9", "so_9")
    channels = [wave.names.index(c) for c in names]
    source = dataset.sources[0]
    files = {}
    for year in (2015, 2018, 2021):
        index = next(
            i for i in range(len(dataset)) if source.time.values[i + 18].year == year
        )
        surface, past, context, truth, forcing, labels = wave.model_sample(
            dataset, [index]
        )
        contexts = torch.stack(
            [advance_season(context, (j + 1) * 5) for j in range(6)], 1
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            initial, _, _ = model.encode_native(surface, past, context, wave.mask)
            final = model.processor.rollout(initial, forcing, contexts)[-1]
            rng = torch.Generator(device=wave.device).manual_seed(4041729)
            members = torch.stack(
                [model.readout(final, wave.mask, rng)[:, -1] for _ in range(8)]
            ).float()
        physical = data.physical(members)[:, 0, channels].cpu().numpy()
        target = data.physical(labels[:, -1])[0, channels].cpu().numpy()
        if not np.isfinite(physical).all():
            raise FloatingPointError("Nonfinite physical endpoint")
        path = output / f"{year}.npz"
        np.savez_compressed(
            path,
            members=physical,
            reference=target,
            channels=np.array(names),
            mask=data.grid["mask"][channels],
            lat=data.grid["lat"],
            lon=data.grid["lon"],
            lead_days=30,
            origin=str(source.time.values[index + 18]),
            target_midpoint=str(source.time.values[index + 24]),
        )
        files[path.name] = digest(path)
        print(json.dumps(dict(event="pilot_export", file=str(path))), flush=True)
    atomic_json(
        dict(files=files, members=8, noise_correlation=model.decoder.noise_correlation),
        output / "COMPLETE.json",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--correlation", type=float, choices=(0.0, 0.5), required=True)
    parser.add_argument("--hours", type=float, default=1.5)
    args = parser.parse_args()
    if not 0 < args.updates <= 2000 or not 0 < args.hours <= 2:
        parser.error("Pilot limited to 2000 updates and two fitting hours")
    args.output.mkdir(parents=True, exist_ok=True)
    data = Samples(args.root / "data/observations", "cuda")
    data.use_observation_normalization()
    model, parent = load_latent_checkpoint(args.checkpoint, data)
    if parent["phase"] != "om4" or parent.get("noise_correlation", 0) != 0:
        raise ValueError("Require the original white-noise OM4 checkpoint")
    for path, key in [
        (args.root / "DATA_READY.json", "data_sha256"),
        (data.root / "SHA256SUMS", "observation_manifest_sha256"),
        (data.root / "statistics.npz", "observation_statistics_sha256"),
    ]:
        if digest(path) != parent[key]:
            raise ValueError("Input contract changed")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    parent_step = saved["state"]["step"]
    if parent_step != 0 or parent["initialization"] != "scratch":
        raise ValueError("Fresh paired pilot requires a scratch step-0 checkpoint")
    del saved
    signature = dict(
        parent,
        producer=os.environ["SAMUDRA_CODE_COMMIT"],
        initialization="scratch-from-shared-step-zero",
        noise_correlation=args.correlation,
        paired_noise_draws=True,
        noise_kernel="5x5 Gaussian sigma=1 grid cell; periodic x, zero-padded y; wet-cell unit variance",
        noise_variance="correlation is filtered variance fraction; rest white; independent channels",
        initial_weights_sha256=digest(args.checkpoint),
        parent_selected_step=parent_step,
        optimizer_restart=True,
        max_updates=args.updates,
        max_seconds=args.hours * 3600,
        scope="One seed, 2000 total OM4-only updates per arm from identical scratch weights; no observation adaptation",
    )
    protocol = args.output / "protocol.json"
    if protocol.exists() and json.loads(protocol.read_text()) != signature:
        raise ValueError("Existing pilot protocol differs")
    atomic_json(signature, protocol)
    loader_args = SimpleNamespace(
        arm="D",
        phase="reconstruction",
        seed=1729,
        readers=16,
        data_root=str(args.root / "data/om4_onedeg_v3"),
        output=str(args.output / "loader"),
        name="correlated-noise-pilot",
        wandb_mode="disabled",
        normalization="instance",
        observation_normalization_root=str(data.root),
        fresh_evolution=True,
        initial_checkpoint=str(args.root / "checkpoints/om4-source/selected.pt"),
        wave1_root="",
        val_origins=12,
        device_cache=True,
        cache_device="cpu",
        device_cache_reserve_gib=32,
        batch_size=1,
    )
    wave = InitializerWave(loader_args)
    try:
        wave.prepare(wave.trainset)
        wave.prepare(wave.valset)
        model.decoder.noise_correlation = args.correlation
        model.decoder.paired_noise_draws = True
        torch.manual_seed(1729)
        model.adapter.requires_grad_(False)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-4
        )

        def objective(step):
            return native_loss(
                model,
                wave,
                wave.trainset,
                batch_at(parent_step + step, len(wave.trainset), 1, 1729),
                1729 + parent_step + step,
            )

        def validate(step):
            loss = sum(
                float(native_loss(model, wave, wave.valset, [i], 101729 + i))
                for i in wave.val_ids
            ) / len(wave.val_ids)
            atomic_json(
                dict(
                    step=step,
                    denoising_loss=loss,
                    note="Within-arm loss; not directly comparable to white-noise denoising loss",
                ),
                args.output / f"validation-{step}.json",
            )
            if step in (1000, 2000):
                atomic_torch(
                    dict(
                        model=model.state_dict(),
                        protocol=dict(signature=signature),
                        state=dict(step=step),
                    ),
                    args.output / f"step-{step}.pt",
                )
            return loss

        state = bounded_fit(
            model,
            optimizer,
            objective,
            validate,
            args.output,
            signature,
            max_updates=args.updates,
            max_seconds=args.hours * 3600,
            checkpoint_every=100,
            validate_every=500,
            emit=wave.emit,
        )
        if not state["complete"] or state["step"] != args.updates:
            raise ValueError("Incomplete fitting invocation")
        atomic_json(
            dict(
                state=state,
                best_checkpoint_sha256=digest(args.output / "best.pt"),
                final_checkpoint_sha256=digest(args.output / "last.pt"),
            ),
            args.output / "PRETRAIN_COMPLETE.json",
        )
        wave.frame_caches.clear()
        wave.args.device_cache = False
        gc.collect()
        source = native_om4_source(
            wave.context_bundle.inference_source,
            LocalLocation(path=Path(loader_args.data_root) / "OM4.zarr"),
            create_rust_io_runtime(4),
        )
        dataset = wave.dataset(source, steps=6)
        wave.prepare(dataset)
        # Fixed final-budget checkpoint, not a test-selected checkpoint.
        export_members(model, wave, dataset, data, args.output / "evaluation/final")
        atomic_json(
            dict(
                state=state,
                signature=signature,
                final_checkpoint_sha256=digest(args.output / "last.pt"),
                exports=["final"],
            ),
            args.output / "PILOT_COMPLETE.json",
        )
    finally:
        if wave.run:
            wave.run.finish()


if __name__ == "__main__":
    main()
