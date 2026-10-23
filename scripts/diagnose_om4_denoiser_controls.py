#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Frozen-model correction, conditioning, solver, exposure and training-length checks."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from diagnose_om4_denoiser_trajectory import (  # type: ignore[import-not-found]
    channel_rmse,
    sha,
)

from samudra.experiments.diffusion_noise import diffusion_noise
from samudra.experiments.monthly_latent import MonthlyLatentOcean
from samudra.experiments.monthly_model_data import ModelMonths, calendar_context


def write(path, data):
    path.write_text(json.dumps(data, allow_nan=False))


def correction_metrics(noisy, prediction, gold, mask, sigma):
    skip = noisy / (1 + sigma.square().reshape(-1, 1, 1, 1))
    learned = (prediction - skip).double() * mask
    desired = (gold - skip).double() * mask
    dot = (learned * desired).sum((-2, -1))
    norm_l = learned.square().sum((-2, -1))
    norm_d = desired.square().sum((-2, -1))
    return dict(
        skip_rmse=channel_rmse(skip, gold, mask),
        prediction_rmse=channel_rmse(prediction, gold, mask),
        learned_rms=channel_rmse(learned, torch.zeros_like(learned), mask),
        correction_cosine=(dot / (norm_l * norm_d).sqrt().clamp_min(1e-20))[0]
        .cpu()
        .tolist(),
        correction_projection=(dot / norm_d.clamp_min(1e-20))[0].cpu().tolist(),
    )


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--checkpoint2k", type=Path, required=True)
    p.add_argument("--checkpoint4k", type=Path, required=True)
    p.add_argument("--subset", type=Path, required=True)
    p.add_argument("--requests", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    started = time.monotonic()
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    grid = dict(np.load(args.bundle / "grid.npz"))
    manifest = json.loads((args.bundle / "COMPLETE.json").read_text())
    expected = {
        "2k": manifest["cases"]["om4"]["checkpoint_sha256"],
        "4k": "2463c9e7e4c49dd02d7aa23b3f9c94ca9dd2b9793a1f10ad5a5e97a61cc0275a",  # pragma: allowlist secret
    }

    def tensor(a):
        return torch.as_tensor(a, device="cuda")[None]

    records, solver, examples = [], [], []
    for label, checkpoint in [("2k", args.checkpoint2k), ("4k", args.checkpoint4k)]:
        assert sha(checkpoint) == expected[label]
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert saved["step"] == int(label[0]) * 1000
        model = MonthlyLatentOcean().cuda().eval()
        model.load_state_dict(saved["ema"], strict=True)
        print(
            json.dumps(
                dict(
                    checkpoint=label,
                    counts=saved.get("counts"),
                    seconds=time.monotonic() - started,
                )
            ),
            flush=True,
        )
        del saved
        with np.load(args.bundle / "om4" / "inputs.npz") as z:
            fields, available, forcing, contexts = (
                tensor(z[k]) for k in ["fields", "available", "forcing", "contexts"]
            )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            initial = model.initialize(
                fields[:, :19], available[:, :19], forcing[:, :19], contexts[:, 18]
            )
            states = model.processor.rollout(initial, forcing[:, 19:], contexts[:, 19:])
            month = model.monthly_condition(
                states[1:],
                torch.as_tensor(
                    manifest["cases"]["om4"]["monthly_weights"], device="cuda"
                ),
            )
        latent_by_stage = dict(initial=initial, day5=states[1], monthly=month)
        for stage, latent in latent_by_stage.items():
            directory = args.bundle / "om4" / stage
            with np.load(directory / "final.npz") as f:
                gold, mask, original_latent = (
                    tensor(f["gold"]).float(),
                    tensor(f["support"]).bool(),
                    tensor(f["latent"]),
                )
            if label == "2k":
                # The original conditioning is retained exactly for the decoder controls.
                latent_delta = float(
                    (latent.float() - original_latent.float()).square().mean().sqrt()
                )
                print(
                    json.dumps(
                        dict(stage=stage, recomputed_latent_rms_difference=latent_delta)
                    ),
                    flush=True,
                )
                latent = original_latent
            meta = json.loads((directory / "trace.json").read_text())
            duration, position = meta["duration_days"], meta["position_days"]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                conditions = {
                    name: model.decoder.condition(h, mask, duration, position)
                    for name, h in [
                        ("correct", latent),
                        ("zero", torch.zeros_like(latent)),
                        ("rolled", torch.roll(latent, shifts=(11, 23), dims=(-2, -1))),
                    ]
                }
            noise = diffusion_noise(
                gold.shape,
                mask,
                torch.Generator(device="cuda").manual_seed(905101),
                model.decoder.core.noise_correlation,
            )
            for call in [0, 16, 24, 28, 32, 36, 40, 44, 48, 56, 62]:
                with np.load(directory / f"call-{call:02d}.npz") as f:
                    sigma = torch.tensor([float(f["sigma"])], device="cuda")
                    sampled = tensor(f["noisy"]).float()
                # Saved sampler inputs belong to 2k only; 4k gold probes are end-to-end.
                sources = [("gold", (gold + sigma[:, None, None, None] * noise) * mask)]
                if label == "2k":
                    sources.append(("sampler", sampled))
                for source, noisy in sources:
                    for condition in ["correct", "zero", "rolled"]:
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            predicted = model.decoder.core(
                                noisy, sigma, conditions[condition], mask
                            )
                        records.append(
                            dict(
                                checkpoint=label,
                                stage=stage,
                                call=call,
                                sigma=float(sigma),
                                source=source,
                                condition=condition,
                                **correction_metrics(
                                    noisy, predicted, gold, mask, sigma
                                ),
                            )
                        )
            write(args.output / "correction.json", records)
            # Same seeded initial noise across solver resolutions. Extra draws at 32.
            settings = (
                [
                    (16, 905001),
                    (32, 905001),
                    (64, 905001),
                    (128, 905001),
                    (32, 905002),
                    (32, 905003),
                ]
                if label == "2k"
                else [(32, 905001), (32, 905002), (32, 905003)]
            )
            for steps, seed in settings:
                trace = []

                def hook(module, inputs, prediction):
                    trace.append(
                        dict(
                            sigma=float(inputs[1][0]),
                            rmse=channel_rmse(prediction, gold, mask),
                        )
                    )

                handle = model.decoder.core.register_forward_hook(hook)
                try:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        prediction = model.decoder.sample(
                            latent,
                            mask,
                            torch.Generator(device="cuda").manual_seed(seed),
                            duration_days=duration,
                            position_days=position,
                            steps=steps,
                        )
                finally:
                    handle.remove()
                solver.append(
                    dict(
                        checkpoint=label,
                        stage=stage,
                        steps=steps,
                        seed=seed,
                        rmse=channel_rmse(prediction, gold, mask),
                        calls=trace,
                    )
                )
                write(args.output / "solver.json", solver)
                print(
                    json.dumps(
                        dict(
                            checkpoint=label,
                            stage=stage,
                            steps=steps,
                            seed=seed,
                            seconds=round(time.monotonic() - started, 1),
                        )
                    ),
                    flush=True,
                )
        del (
            fields,
            available,
            forcing,
            contexts,
            initial,
            states,
            latent_by_stage,
            month,
        )
        reader = ModelMonths(args.subset / "OM4.zarr", "om4", grid, args.subset)
        for request in json.loads(args.requests.read_text()):
            ix = request["indices"]
            # Assert every requested chunk exists: Zarr must not silently fill missing input.
            for name in reader.names + ["tauuo", "tauvo", "hfds"]:
                for i in ix:
                    assert (args.subset / "OM4.zarr" / name / f"{i}.0.0").is_file()
            raw = np.stack([reader.store[n][ix] for n in reader.names], 1)
            assert np.isfinite(raw[:, reader.mask]).all()
            fields = tensor(
                np.where(
                    reader.mask,
                    (raw - grid["mean"][:, None, None]) / grid["std"][:, None, None],
                    0,
                ).astype("float32")
            )
            mask = tensor(reader.mask)
            available = torch.zeros_like(fields, dtype=torch.bool)
            available[:, :, [38, 76]] = mask[:, None, [38, 76]]
            raw_forcing = np.stack(
                [reader.store[n][ix] for n in ["tauuo", "tauvo", "hfds"]], 1
            )
            assert np.isfinite(raw_forcing[:, :, reader.mask[38]]).all()
            forcing = tensor(
                np.where(
                    reader.mask[38],
                    (raw_forcing - reader.forcing_mean[:, None, None])
                    / reader.forcing_std[:, None, None],
                    0,
                ).astype("float32")
            )
            contexts = tensor(
                calendar_context(
                    reader.lat, reader.lon, [reader.midpoints[i] for i in ix]
                )
            )
            gold = fields[:, -1]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                latent = model.initialize(fields, available, forcing, contexts[:, -1])
                condition = model.decoder.condition(latent, mask, 5.0, -2.5)
            noise = diffusion_noise(
                gold.shape,
                mask,
                torch.Generator(device="cuda").manual_seed(905101),
                model.decoder.core.noise_correlation,
            )
            curve = []
            for sigma_value in [
                80.0,
                16.5914,
                6.4637,
                2.1739,
                1.1743,
                0.5977,
                0.1226,
                0.002,
            ]:
                sigma = torch.tensor([sigma_value], device="cuda")
                noisy = (gold + sigma[:, None, None, None] * noise) * mask
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    predicted = model.decoder.core(noisy, sigma, condition, mask)
                curve.append(
                    dict(sigma=sigma_value, rmse=channel_rmse(predicted, gold, mask))
                )
            trace = []

            def hook(module, inputs, prediction):
                trace.append(
                    dict(
                        sigma=float(inputs[1][0]),
                        rmse=channel_rmse(prediction, gold, mask),
                    )
                )

            handle = model.decoder.core.register_forward_hook(hook)
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction = model.decoder.sample(
                        latent,
                        mask,
                        torch.Generator(device="cuda").manual_seed(905001),
                        position_days=-2.5,
                        steps=32,
                    )
            finally:
                handle.remove()
            examples.append(
                dict(
                    checkpoint=label,
                    **request,
                    gold_curve=curve,
                    sampler_curve=trace,
                    final_rmse=channel_rmse(prediction, gold, mask),
                )
            )
            write(args.output / "examples.json", examples)
            print(
                json.dumps(
                    dict(
                        checkpoint=label,
                        example=request["name"],
                        seconds=round(time.monotonic() - started, 1),
                    )
                ),
                flush=True,
            )
        del model
    write(
        args.output / "COMPLETE.json",
        dict(
            checkpoints=expected,
            seconds=time.monotonic() - started,
            torch=torch.__version__,
            gpu=torch.cuda.get_device_name(),
            source_sha256=sha(Path(__file__)),
            correction_records=len(records),
            solver_runs=len(solver),
            example_runs=len(examples),
        ),
    )


if __name__ == "__main__":
    main()
