#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Five-arm pilot runner; requires a full, producer-matched qualification marker."""

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from samudra.experiments.monthly_latent import MonthlyLatentOcean
from samudra.experiments.monthly_model_data import ModelMonths, model_loss, task_source
from samudra.experiments.monthly_pilot_data import observation_loss
from samudra.experiments.monthly_pilot_evaluation import evaluate
from samudra.experiments.observation_metrics import selection_score
from samudra.experiments.observation_training import Samples


def atomic_save(value, path):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=list("ABCDE"), required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--observations", required=True)
    p.add_argument("--om4", required=True)
    p.add_argument("--cm4", required=True)
    p.add_argument("--producer", required=True)
    p.add_argument("--qualification", required=True)
    args = p.parse_args()
    proof = json.loads(Path(args.qualification).read_text())
    if (
        proof.get("producer") != args.producer
        or proof.get("production_qualified") is not True
    ):
        raise ValueError("Missing full producer-matched qualification")
    if not proof.get("data_contracts"):
        raise ValueError("Missing qualified data contracts")
    for path, expected in proof["data_contracts"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError("Qualified data contract changed: " + path)
    out = Path(args.root) / args.arm
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.manual_seed(1729)
    torch.cuda.manual_seed_all(1729)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    data = Samples(
        args.observations,
        "cuda",
        surface_fill="zero",
        global_observations=True,
        cache_size=256,
    )
    data.cache_prepared = True
    stores = {}
    if args.arm != "E":
        stores["om4"] = ModelMonths(
            Path(args.om4) / "OM4.zarr",
            "om4",
            data.grid,
            args.om4,
            packed=proof.get("packed_om4"),
        )
    if args.arm in "CD":
        stores["cm4"] = ModelMonths(args.cm4, "cm4", data.grid, args.om4)
    model = MonthlyLatentOcean().cuda()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    generator = torch.Generator(device="cuda").manual_seed(1729)
    rng = np.random.default_rng(1729)
    paths, validation = data.paths("train"), data.paths("validation")
    protocol = dict(
        arm=args.arm,
        producer=args.producer,
        seed=1729,
        updates=2000,
        cooldown=1000,
        effective_batch=8,
        lr=1e-4,
        final_lr=1e-6,
        ema_half_life=100,
        validation_every=250,
        members=8,
        sampler_steps=32,
        qualification_sha256=hashlib.sha256(
            Path(args.qualification).read_bytes()
        ).hexdigest(),
        observations=args.observations,
        om4=args.om4,
        cm4=args.cm4,
    )
    manifest = out / "protocol.json"
    if manifest.exists() and json.loads(manifest.read_text()) != protocol:
        raise ValueError("Frozen protocol differs")
    manifest.write_text(json.dumps(protocol, indent=2) + "\n")
    import wandb

    run = wandb.init(
        entity="ocean_emulators",
        project="observational-transfer",
        name="monthly-latent-" + args.arm,
        id=hashlib.sha256(str(out).encode()).hexdigest()[:12],
        resume="allow",
        dir=str(out),
        mode="online",
        config=protocol,
    )
    if not run.url:
        raise ValueError("No online W&B URL")
    (out / "wandb-url.txt").write_text(run.url + "\n")
    print(json.dumps(dict(wandb_url=run.url)), flush=True)
    step, best = 0, math.inf
    counts = dict(obs=0, om4=0, cm4=0)
    state_path = out / "last.pt"
    if state_path.exists():
        saved = torch.load(state_path, map_location="cuda", weights_only=False)
        if saved["protocol"] != protocol:
            raise ValueError("Resume contract differs")
        model.load_state_dict(saved["model"])
        ema.load_state_dict(saved["ema"])
        optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["generator"].cpu())
        rng.bit_generator.state = saved["rng"]
        torch.set_rng_state(saved["cpu_rng"].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda_rng"]])
        step, best, counts = saved["step"], saved["best"], saved["counts"]
    control = evaluate(ema, data, validation, climatology=True)

    def save(path):
        atomic_save(
            dict(
                protocol=protocol,
                model=model.state_dict(),
                ema=ema.state_dict(),
                optimizer=optimizer.state_dict(),
                generator=generator.get_state(),
                rng=rng.bit_generator.state,
                cpu_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all(),
                step=step,
                best=best,
                counts=counts,
            ),
            path,
        )

    while step < 3000:
        if step == 2000:
            ema.load_state_dict(model.state_dict())
        cooldown = step >= 2000
        source = task_source(args.arm, step, cooldown)
        lr = 1e-4 if not cooldown else 1e-4 + (1e-6 - 1e-4) * (step - 1999) / 1000
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        optimizer.zero_grad(set_to_none=True)
        started = time.monotonic()
        losses = []
        for _ in range(8):
            draw = rng.random()
            task = (
                "surface"
                if cooldown or args.arm not in "BD" or draw < 0.5
                else "variable"
                if draw < 0.75
                else "gap"
            )
            if source == "obs":
                sample = data.load(paths[int(rng.integers(len(paths)))])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, _ = observation_loss(model, data, sample, task, generator)
            else:
                reader = stores[source]
                batch = reader.sample(int(rng.integers(len(reader.windows))), "cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model_loss(model, batch, task, generator, data.area)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training objective")
            (loss / 8).backward()
            losses.append(float(loss.detach()))
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        with torch.no_grad():
            for target, current in zip(
                ema.parameters(), model.parameters(), strict=True
            ):
                target.lerp_(current, 1 - 2 ** (-1 / 100))
        step += 1
        counts[source] += 1
        record = dict(
            step=step,
            source=source,
            cooldown=cooldown,
            loss=float(np.mean(losses)),
            gradient_norm=float(norm),
            lr=lr,
            seconds=time.monotonic() - started,
        )
        if step % 250 == 0:
            metrics = evaluate(ema, data, validation)
            value = selection_score(metrics, control, [], metric="rmse-only")
            record["validation_rmse_blend"] = value
            (out / f"validation-{step:05d}.json").write_text(
                json.dumps(metrics, indent=2) + "\n"
            )
            if value < best:
                best = value
                save(out / "best.pt")
            if step in (2000, 3000):
                save(out / f"endpoint-{step:05d}.pt")
        if step % 50 == 0:
            save(state_path)
        with (out / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        run.log(record, step=step)
    save(state_path)
    (out / "TRAIN_COMPLETE.json").write_text(
        json.dumps(
            dict(step=step, counts=counts, best=best, producer=args.producer), indent=2
        )
        + "\n"
    )
    run.finish()


if __name__ == "__main__":
    main()
