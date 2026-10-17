#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Matched clipping and OM4-adapter continuations from the qualified A 2k state."""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from samudra.experiments.monthly_forcing_adapter import ModelForcingAdapter
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
    p.add_argument("--arm", choices=["clip1", "clip5", "adapter"], required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--observations", required=True)
    p.add_argument("--om4", required=True)
    p.add_argument("--cm4", required=True)
    p.add_argument("--producer", required=True)
    p.add_argument("--qualification", required=True)
    p.add_argument("--source-checkpoint", required=True)
    p.add_argument("--source-sha256", required=True)
    p.add_argument("--qualify", action="store_true")
    args = p.parse_args()
    if (
        hashlib.sha256(Path(args.source_checkpoint).read_bytes()).hexdigest()
        != args.source_sha256
    ):
        raise ValueError("Source checkpoint hash mismatch")
    clip_threshold = 5.0 if args.arm == "clip5" else 1.0
    proof = json.loads(Path(args.qualification).read_text())
    if (
        proof.get("producer") != "1c9817d44"
        or proof.get("production_qualified") is not True
    ):
        raise ValueError("Missing full producer-matched qualification")
    if not proof.get("data_contracts"):
        raise ValueError("Missing qualified data contracts")
    for path, expected in proof["data_contracts"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError("Qualified data contract changed: " + path)
    out = (
        Path(args.root) / ("qualification" if args.qualify else "production") / args.arm
    )
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
    stores["om4"] = ModelMonths(
        Path(args.om4) / "OM4.zarr",
        "om4",
        data.grid,
        args.om4,
        packed=proof.get("packed_om4"),
    )
    model = MonthlyLatentOcean().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    source_state = torch.load(
        args.source_checkpoint, map_location="cpu", weights_only=False
    )
    if source_state["step"] != 2000 or source_state["protocol"]["arm"] != "A":
        raise ValueError("Expected original A main endpoint at 2000")
    model.load_state_dict(source_state["model"])
    optimizer.load_state_dict(source_state["optimizer"])
    forcing_adapter = ModelForcingAdapter().cuda() if args.arm == "adapter" else None
    if forcing_adapter is not None:
        model.add_module("model_forcing_adapter", forcing_adapter)
        optimizer.add_param_group(
            {"params": list(forcing_adapter.parameters()), "lr": 1e-4}
        )
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    ema_state = dict(source_state["ema"])
    if forcing_adapter is not None:
        ema_state.update(
            {
                "model_forcing_adapter." + k: v
                for k, v in forcing_adapter.state_dict().items()
            }
        )
    ema.load_state_dict(ema_state)
    for name, value in source_state["model"].items():
        assert torch.equal(model.state_dict()[name].cpu(), value), name
    generator = torch.Generator(device="cuda").manual_seed(1729)
    rng = np.random.default_rng(1729)
    paths, validation = data.paths("train"), data.paths("validation")
    protocol = dict(
        arm=args.arm,
        producer=args.producer,
        seed=1729,
        start=2000,
        updates=1000,
        cooldown=0,
        source_sha256=args.source_sha256,
        clip_threshold=clip_threshold,
        model_forcing_adapter=args.arm == "adapter",
        qualify=args.qualify,
        effective_batch=8,
        lr=1e-4,
        final_lr=1e-4,
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
    if not args.qualify:
        qualified = json.loads(
            (
                Path(args.root) / "qualification" / args.arm / "QUALIFIED.json"
            ).read_text()
        )
        if (
            qualified["producer"] != args.producer
            or qualified["source_sha256"] != args.source_sha256
        ):
            raise ValueError("Qualification producer/source mismatch")
    import wandb

    run = wandb.init(
        entity="ocean_emulators",
        project="observational-transfer",
        name="monthly-clip-adapter-" + args.arm,
        id=hashlib.sha256(str(out).encode()).hexdigest()[:12],
        resume="allow",
        dir=str(out),
        mode="disabled" if args.qualify else "online",
        config=protocol,
    )
    if not args.qualify and not run.url:
        raise ValueError("No online W&B URL")
    if run.url:
        (out / "wandb-url.txt").write_text(run.url + "\n")
    print(json.dumps(dict(wandb_url=run.url)), flush=True)
    step, best = 2000, source_state["best"]
    counts = dict(source_state["counts"])
    generator.set_state(source_state["generator"].cpu())
    rng.bit_generator.state = source_state["rng"]
    torch.set_rng_state(source_state["cpu_rng"].cpu())
    torch.cuda.set_rng_state_all([v.cpu() for v in source_state["cuda_rng"]])
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

    if step == 2000:
        save(out / "fork-02000.pt")
        save(out / "best.pt")
    while step < (2002 if args.qualify else 3000):
        cooldown = False
        source = task_source("A", step, cooldown)
        lr = 1e-4
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        optimizer.zero_grad(set_to_none=True)
        started = time.monotonic()
        losses = []
        for _ in range(8):
            rng.random()  # Preserve the original A task-draw stream.
            task = "surface"
            if source == "obs":
                sample = data.load(paths[int(rng.integers(len(paths)))])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, _ = observation_loss(model, data, sample, task, generator)
            else:
                reader = stores[source]
                batch = reader.sample(int(rng.integers(len(reader.windows))), "cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if forcing_adapter is not None:
                        batch = dict(batch)
                        original_forcing = batch["forcing"]
                        batch["forcing"] = forcing_adapter(
                            original_forcing, batch["mask"][38]
                        )
                        if step == 2001:
                            torch.testing.assert_close(
                                batch["forcing"], original_forcing, rtol=0, atol=0
                            )
                    loss = model_loss(model, batch, task, generator, data.area)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training objective")
            (loss / 8).backward()
            losses.append(float(loss.detach()))
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), clip_threshold, error_if_nonfinite=True
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
        # Sparse per-source series retain the common optimizer-update axis.
        # Missing sources are omitted, never logged as zero-valued losses.
        record.update(
            {
                f"train/{source}/loss": record["loss"],
                f"train/{source}/gradient_norm_preclip": record["gradient_norm"],
                f"train/{source}/clip_scale": min(
                    1.0, clip_threshold / (float(norm) + 1e-6)
                ),
                f"train/{source}/clipped": float(norm) > clip_threshold,
            }
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
    if args.qualify:
        reloaded = torch.load(state_path, map_location="cuda", weights_only=False)
        for key in ("model", "ema"):
            live = model.state_dict() if key == "model" else ema.state_dict()
            for name, value in live.items():
                assert torch.equal(value, reloaded[key][name]), (key, name)
        assert reloaded["step"] == 2002
        if forcing_adapter is not None:
            assert not torch.equal(forcing_adapter.weight.detach().cpu(), torch.eye(3))
        (out / "QUALIFIED.json").write_text(
            json.dumps(
                dict(
                    producer=args.producer,
                    source_sha256=args.source_sha256,
                    step=step,
                    finite_two_task_updates=True,
                )
            )
        )
    run.finish()


if __name__ == "__main__":
    main()
