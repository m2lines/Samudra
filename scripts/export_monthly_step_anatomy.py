#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Export fixed A examples and actual denoiser calls without altering sampling."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from samudra.experiments.monthly_latent import MonthlyLatentOcean
from samudra.experiments.monthly_model_data import ModelMonths
from samudra.experiments.monthly_pilot_data import observation_batch
from samudra.experiments.observation_pilot import atomic_json, digest
from samudra.experiments.observation_training import Samples


def array(value):
    return (
        value.detach().float().cpu().numpy()
        if torch.is_tensor(value)
        else np.asarray(value)
    )


def save(path, **values):
    values = {k: array(v) for k, v in values.items()}
    np.savez_compressed(path, **values)
    with np.load(path) as check:
        for key, value in values.items():
            np.testing.assert_array_equal(check[key], value)


@torch.no_grad()
def trace(model, latent, support, target, directory, duration=5.0, position=0.0):
    directory.mkdir()
    support = support.bool().expand(1, 77, 180, 360)
    records: list[dict] = []

    def hook(module, args, output):
        call = len(records)
        sigma = float(args[1][0])
        # Both the predictor and Heun corrector calls are retained.
        save(
            directory / f"call-{call:02d}.npz",
            noisy=args[0][0],
            clean=output[0],
            sigma=sigma,
        )
        records.append(
            dict(
                call=call,
                sigma=sigma,
                role="predictor" if call % 2 == 0 else "corrector",
            )
        )

    handle = model.decoder.core.register_forward_hook(hook)
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model.decoder.sample(
                latent,
                support,
                torch.Generator(device="cuda").manual_seed(905001),
                steps=32,
                duration_days=duration,
                position_days=position,
            )
    finally:
        handle.remove()
    # Instrumentation must be passive: rerun the untouched sampler with identical RNG.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        reference = model.decoder.sample(
            latent,
            support,
            torch.Generator(device="cuda").manual_seed(905001),
            steps=32,
            duration_days=duration,
            position_days=position,
        )
    torch.testing.assert_close(output, reference, rtol=0, atol=0)
    assert len(records) == 63
    save(
        directory / "final.npz",
        prediction=output[0],
        gold=target[0],
        support=support[0],
        latent=latent[0],
    )
    atomic_json(
        dict(
            calls=records,
            steps=32,
            seed=905001,
            duration_days=duration,
            position_days=position,
            instrumented_equals_original=True,
        ),
        directory / "trace.json",
    )


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    data = Samples(
        config["samples"], "cuda", surface_fill="zero", global_observations=True
    )
    save(args.output / "grid.npz", **data.grid)
    reader = ModelMonths(
        Path(config["om4"]) / "OM4.zarr", "om4", data.grid, Path(config["om4"])
    )
    # Training-window reader intentionally excludes held-out dates. Read matched
    # January 2015 through the audited prepared annual payload instead.
    with np.load(config["prepared"]) as z:
        prepared = dict(z)
    from samudra.experiments.monthly_model_annual import model_batch

    om4 = model_batch(data, prepared)
    indices = prepared["source_indices"][:26]
    full = np.stack([reader.store[n][indices] for n in reader.names], 1)
    normalized = np.where(
        reader.mask,
        (full - data.grid["mean"][:, None, None]) / data.grid["std"][:, None, None],
        0,
    ).astype("float32")
    np.testing.assert_allclose(normalized[:19], prepared["fields"], rtol=0, atol=0)
    sample = data.load(data.root / "test" / "2015-01.npz")
    obs = observation_batch(
        data, sample, "surface", torch.Generator(device="cuda").manual_seed(1)
    )
    manifest = dict(
        origin="2015-01-01",
        producer=os.environ["SAMUDRA_CODE_COMMIT"],
        config_sha256=digest(args.config),
        cases={},
    )
    for case, source, step in [
        ("om4", "om4", 2000),
        ("obs_pre", "obs", 2000),
        ("obs", "obs", 3000),
    ]:
        checkpoint_key = "om4" if step == 2000 else "obs"
        checkpoint = Path(config[checkpoint_key + "_checkpoint"])
        expected = config[checkpoint_key + "_sha256"]
        assert digest(checkpoint) == expected
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert saved["step"] == step
        model = MonthlyLatentOcean().cuda().eval()
        model.load_state_dict(saved["ema"], strict=True)
        dest = args.output / case
        dest.mkdir()
        batch = om4 if source == "om4" else obs
        if source == "om4":
            forcing, contexts = om4["forcing"], om4["contexts"]
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                forcing = model.adapt(sample["atmosphere"])
            contexts = sample["contexts"]
        # Use exactly the monthly training trajectory, including the final bin
        # straddling February, so the monthly auxiliary view is directly comparable.
        weights = sample["month_weights"]
        length = 19 + len(weights)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            states = model.trajectory(
                batch["fields"][:, :length],
                batch["available"][:, :length],
                forcing[:, :length],
                contexts[:, :length],
            )
        save(
            dest / "inputs.npz",
            fields=batch["fields"][0, :19],
            available=batch["available"][0, :19],
            forcing=forcing[0, :length],
            contexts=contexts[0, :length],
            latents=torch.stack(states)[:, 0],
            om4_context=normalized[[18, 19]],
            forcing_mean=reader.forcing_mean,
            forcing_std=reader.forcing_std,
        )
        if source == "obs":
            save(
                dest / "atmosphere.npz",
                normalized=sample["atmosphere"][0],
                mean=data.stats["atmosphere_mean"],
                std=data.stats["atmosphere_std"],
            )
        targets = []
        supports = []
        for index in (18, 19):
            if source == "om4":
                targets.append(data.tensor(normalized[index])[None])
                supports.append(data.mask.bool()[None])
            else:
                target = torch.zeros_like(obs["target"])
                mask = torch.zeros_like(obs["support"])
                target[:, [38, 76]] = sample["surface"][:, index]
                mask[:, [38, 76]] = sample["validity"][:, index].bool()
                targets.append(target)
                supports.append(mask)
        for name, state, target, support, position in zip(
            ("initial", "day5"), states[:2], targets, supports, (-2.5, 2.5), strict=True
        ):
            trace(model, state, support, target, dest / name, position=position)
            print(json.dumps(dict(case=case, stage=name)), flush=True)
            if source == "obs":
                context_index = 18 if name == "initial" else 19
                trace(
                    model,
                    state,
                    data.mask.bool()[None],
                    data.tensor(normalized[context_index])[None],
                    dest / (name + "-full"),
                    position=position,
                )
        monthly = model.monthly_condition(states[1:], weights)
        if source == "om4":
            target = (
                data.tensor(normalized[19:length])[None]
                * weights[None, :, None, None, None]
            ).sum(1)
            support = data.mask.bool()[None]
        else:
            target, support = obs["target"], obs["support"]
        trace(
            model,
            monthly,
            support,
            target,
            dest / "monthly",
            duration=31.0,
            position=15.5,
        )
        manifest["cases"][case] = dict(
            checkpoint=str(checkpoint),
            checkpoint_sha256=expected,
            step=step,
            weights="ema",
            counts=saved["counts"],
            monthly_weights=array(weights).tolist(),
            initial_inputs="19 five-day SST/SSH frames only",
            source=source,
        )
        del model
    manifest["files"] = {
        str(f.relative_to(args.output)): digest(f)
        for f in args.output.rglob("*")
        if f.is_file()
    }
    atomic_json(manifest, args.output / "COMPLETE.json")


if __name__ == "__main__":
    main()
