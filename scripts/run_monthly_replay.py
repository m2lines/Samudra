#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Matched longer-training and from-scratch observation replay experiments."""

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
from samudra.experiments.monthly_replay import (
    ReplayIntervals,
    ReplayPool,
    replay_loss,
    window_layout,
)
from samudra.experiments.observation_metrics import selection_score
from samudra.experiments.observation_training import Samples


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_save(value, path):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def assert_exact(left, right, path="state"):
    """Compare optimizer moments, latent pool, EMA and RNG, not only weights."""
    if isinstance(left, torch.Tensor):
        if not torch.equal(left.cpu(), right.cpu()):
            raise ValueError("Serialized tensor changed: " + path)
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise ValueError("Serialized keys changed: " + path)
        for key in left:
            assert_exact(left[key], right[key], path + "/" + str(key))
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise ValueError("Serialized length changed: " + path)
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            assert_exact(a, b, path + "/" + str(index))
    elif left != right:
        raise ValueError("Serialized value changed: " + path)


def phase_settings(step, mode, arm):
    main = 4000 if mode in ("extend", "fresh-4k") else 2000
    if step < (2000 if mode == "extend" else 0) or step >= main + 1000:
        raise ValueError("Update outside authorized experiment")
    cooldown = step >= main
    lr = 1e-4 if not cooldown else 1e-4 + (1e-6 - 1e-4) * (step - main + 1) / 1000
    return cooldown, task_source(arm, step, cooldown), lr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--qualify", action="store_true")
    a = p.parse_args()
    config = json.loads((a.root / "paths.json").read_text())
    row = next(r for r in config["runs"] if r["name"] == a.name)
    replay = row["mode"] == "replay"
    fresh = replay or row["mode"] == "fresh-4k"
    architecture = row.get("architecture", {})
    if architecture and architecture != {"latent_width": 64, "latent_stride": 1}:
        raise ValueError("Unqualified architecture requested")
    source_path = Path(row["checkpoint"])
    if digest(source_path) != row["sha256"]:
        raise ValueError("Source checkpoint changed")
    legacy = json.loads(Path(config["legacy_qualification"]).read_text())
    if legacy["producer"] != "1c9817d44" or not legacy["production_qualified"]:
        raise ValueError("Original data/runtime qualification missing")
    for path, sha in legacy["data_contracts"].items():
        if digest(path) != sha:
            raise ValueError("Original data contract changed: " + path)
    replay_proof = None
    if replay:
        replay_proof = json.loads(
            (Path(config["replay_cache"]) / "COMPLETE.json").read_text()
        )
        if replay_proof["exact_original_months"] != 243:
            raise ValueError("Replay cache not qualified against original observations")
        for path, sha in replay_proof["contracts"].items():
            if digest(path) != sha:
                raise ValueError("Replay normalization/split contract changed")
        # Qualification performs the expensive full read-back; production pins
        # that proof and rechecks file sizes and the small time index.
        for name, sha in replay_proof["files"].items():
            if (a.qualify or name != "values.npy") and digest(
                Path(config["replay_cache"]) / name
            ) != sha:
                raise ValueError("Replay cache read-back mismatch")
        if (Path(config["replay_cache"]) / "values.npy").stat().st_size != replay_proof[
            "bytes"
        ]:
            raise ValueError("Replay cache size changed")
    out = a.root / ("qualification" if a.qualify else "production") / a.name
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        **row,
        producer=config["producer"],
        config_sha256=digest(a.root / "paths.json"),
        qualification=a.qualify,
        seed=1729,
        effective_batch=8,
        new_main_updates=4000 if row["mode"] == "fresh-4k" else 2000,
        cooldown=1000,
        main_end=2000 if replay else 4000,
        lr=1e-4,
        final_lr=1e-6,
        clip=1.0,
        ema_half_life=100,
        replay_fraction_of_obs=0.5 if replay else 0,
        replay_max_age=250,
        replay_entries_per_lead_month=8,
        replay_max_steps=73,
        latent_magnitude_penalty=0,
        legacy_qualification_sha256=digest(config["legacy_qualification"]),
        replay_proof_sha256=digest(Path(config["replay_cache"]) / "COMPLETE.json")
        if replay
        else None,
        selection="original nine-origin monthly five-component rmse-only blend",
    )
    manifest = out / "protocol.json"
    if manifest.exists() and json.loads(manifest.read_text()) != protocol:
        raise ValueError("Frozen protocol changed")
    manifest.write_text(json.dumps(protocol, indent=2) + "\n")
    if not a.qualify:
        proof = json.loads(
            (a.root / "qualification" / a.name / "QUALIFIED.json").read_text()
        )
        expected_protocol = {**protocol, "qualification": True}
        if (
            proof["protocol"] != expected_protocol
            or not proof["exact_serialized_update"]
        ):
            raise ValueError("Producer-matched qualification missing")
    torch.set_num_threads(8)
    torch.manual_seed(1729)
    torch.cuda.manual_seed_all(1729)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    data = Samples(
        config["samples"],
        "cuda",
        surface_fill="zero",
        global_observations=True,
        cache_size=256,
    )
    data.cache_prepared = True
    reader = None
    if row["arm"] == "A":
        reader = ModelMonths(
            Path(config["om4"]) / "OM4.zarr",
            "om4",
            data.grid,
            config["om4"],
            packed=legacy["packed_om4"],
        )
    model = MonthlyLatentOcean(**architecture).cuda()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    generator = torch.Generator(device="cuda").manual_seed(1729)
    rng = np.random.default_rng(1729)
    replay_rng = np.random.default_rng(1729001)
    pool = ReplayPool()
    intervals = ReplayIntervals(config["replay_cache"]) if replay else None
    initial_digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        initial_digest.update(name.encode())
        initial_digest.update(value.detach().cpu().numpy().tobytes())
    initial_hash = initial_digest.hexdigest()
    step, best = 0, math.inf
    counts = dict(obs=0, om4=0, cm4=0)
    replay_examples, fresh_examples = 0, 0

    def restore(saved, *, continuation=False):
        nonlocal step, best, counts, replay_examples, fresh_examples
        model.load_state_dict(saved["model"])
        ema.load_state_dict(saved["ema"])
        optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["generator"].cpu())
        rng.bit_generator.state = saved["rng"]
        torch.set_rng_state(saved["cpu_rng"].cpu())
        torch.cuda.set_rng_state_all([v.cpu() for v in saved["cuda_rng"]])
        step, counts = saved["step"], saved["counts"]
        if not continuation:
            best = saved["best"]
            replay_rng.bit_generator.state = saved["replay_rng"]
            pool.load_state_dict(saved["replay_pool"])
            replay_examples, fresh_examples = (
                saved["replay_examples"],
                saved["fresh_examples"],
            )

    if not fresh:
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        if (
            source["step"] != 2000
            or source["protocol"]["arm"] != row["arm"]
            or source["protocol"]["seed"] != 1729
        ):
            raise ValueError("Source arm/step/seed differs")
        restore(source, continuation=True)
        # Loading is exact; no qualification weights or cooled optimizer state.
        for name, value in source["model"].items():
            if not torch.equal(model.state_dict()[name].cpu(), value):
                raise ValueError("Checkpoint transfer changed model state")
        del source
    state_path = out / "last.pt"
    if state_path.exists():
        saved = torch.load(state_path, map_location="cpu", weights_only=False)
        if saved["protocol"] != protocol:
            raise ValueError("Resume protocol differs")
        restore(saved)
        del saved
    train, validation = data.paths("train"), data.paths("validation")
    control = evaluate(ema, data, validation, climatology=True)
    main_end = protocol["main_end"]
    end = main_end + 1000

    def save(path):
        atomic_save(
            dict(
                protocol=protocol,
                model=model.state_dict(),
                ema=ema.state_dict(),
                optimizer=optimizer.state_dict(),
                generator=generator.get_state(),
                rng=rng.bit_generator.state,
                replay_rng=replay_rng.bit_generator.state,
                replay_pool=pool.state_dict(),
                cpu_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all(),
                step=step,
                best=best,
                counts=counts,
                replay_examples=replay_examples,
                fresh_examples=fresh_examples,
            ),
            path,
        )

    def update():
        nonlocal step, replay_examples, fresh_examples
        if step == main_end:
            ema.load_state_dict(model.state_dict())
        cooldown, source, lr = phase_settings(step, row["mode"], row["arm"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses, lead_days, ages, components, rms = [], [], [], [], []
        started = time.monotonic()
        for micro in range(8):
            rng.random()  # Preserve the original task-draw stream.
            if source == "obs":
                path = train[int(rng.integers(len(train)))]
                entry = pool.draw(replay_rng, step) if replay and micro % 2 else None
                captured = []

                def capture(states):
                    origin = entry["origin"] if entry is not None else path.stem
                    month = entry["month"] if entry is not None else path.stem
                    if replay:
                        pool.add_next(
                            origin,
                            month,
                            states,
                            step,
                            origin_step=None if entry is None else entry["origin_step"],
                        )
                    captured.append(
                        float(states[-1].detach().float().square().mean().sqrt())
                    )

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if entry is None:
                        sample = data.load(path)
                        loss, pieces = observation_loss(
                            model, data, sample, "surface", generator, capture=capture
                        )
                        fresh_examples += 1
                        lead_days.append(0)
                    else:
                        loss, pieces = replay_loss(
                            model, data, intervals, entry, generator, capture=capture
                        )
                        replay_examples += 1
                        lead_days.append(entry["bin_index"] * 5)
                        ages.append(step - entry["origin_step"])
                components.append(pieces)
                rms.extend(captured)
            else:
                if reader is None:
                    raise ValueError("Model data missing")
                batch = reader.sample(int(rng.integers(len(reader.windows))), "cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model_loss(model, batch, "surface", generator, data.area)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
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
        record.update(
            {
                f"train/{source}/loss": record["loss"],
                f"train/{source}/gradient_norm_preclip": float(norm),
                f"train/{source}/clip_scale": min(1, 1 / (float(norm) + 1e-6)),
            }
        )
        if source == "obs":
            record.update(
                {
                    f"train/obs/{key}": float(np.mean([c[key] for c in components]))
                    for key in components[0]
                }
            )
            record.update(
                replay_examples=replay_examples,
                fresh_examples=fresh_examples,
                replay_start_lead_days_mean=float(np.mean(lead_days)),
                replay_start_lead_days_max=max(lead_days),
                replay_age_max=max(ages, default=0),
                latent_rms_mean=float(np.mean(rms)),
            )
        return record

    if a.qualify:
        # Exercise both sources and actual replay, then compare an entire
        # serialized optimizer update including pool and all random streams.
        records = [update() for _ in range(4)]
        if replay and replay_examples == 0:
            raise ValueError("No real observation replay was exercised")
        save(out / "serialization-source.pt")
        first = update()
        save(out / "serialization-expected.pt")
        restore(
            torch.load(
                out / "serialization-source.pt", map_location="cpu", weights_only=False
            )
        )
        second = update()
        for key in set(first) - {"seconds"}:
            if first[key] != second[key]:
                raise ValueError(f"Serialized update metric differs: {key}")
        save(out / "serialization-replayed.pt")
        assert_exact(
            torch.load(
                out / "serialization-expected.pt",
                map_location="cpu",
                weights_only=False,
            ),
            torch.load(
                out / "serialization-replayed.pt",
                map_location="cpu",
                weights_only=False,
            ),
        )
        longest = None
        if replay:
            assert intervals is not None
            # Exercise the largest supported lead, without promoting these
            # diagnostic states or weights into the production run.
            from samudra.experiments.monthly_pilot_data import observation_batch

            origin, month = "2001-05-01", "2002-04"
            start_bin, end_bin, *_ = window_layout(origin, month)
            sample = data.load(data.root / "train/2001-05.npz")
            batch = observation_batch(data, sample, "surface", generator)
            import pandas as pd

            prefix = intervals.load(
                data, pd.date_range(origin, periods=start_bin, freq="5D")
            )
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                initial = model.initialize(
                    batch["fields"][:, :19],
                    batch["available"][:, :19],
                    model.adapt(sample["atmosphere"][:, :19]),
                    sample["contexts"][:, 18],
                )
                state = model.processor.rollout(
                    initial, model.adapt(prefix["atmosphere"]), prefix["contexts"]
                )[-1]
            entry = dict(
                origin=origin, month=month, bin_index=start_bin, latent=state.cpu()
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = replay_loss(model, data, intervals, entry, generator)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1, error_if_nonfinite=True
            )
            longest = dict(
                final_bin_day=end_bin * 5,
                loss=float(loss.detach()),
                gradient_norm=float(norm),
            )
            if end_bin != 73 or not math.isfinite(longest["loss"]):
                raise ValueError("Long-lead replay qualification failed")
        metrics = evaluate(ema, data, validation[:1])
        if not math.isfinite(selection_score(metrics, control, [], metric="rmse-only")):
            raise ValueError("Invalid diffusion evaluation")
        (out / "QUALIFIED.json").write_text(
            json.dumps(
                dict(
                    protocol=protocol,
                    exact_serialized_update=True,
                    initial_random_backbone_sha256=initial_hash,
                    records=records,
                    replay_examples=replay_examples,
                    peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                    parameters=sum(p.numel() for p in model.parameters()),
                    latent_shape=list(model.latent_shape),
                    latent_width=model.latent_width,
                    source_loading_exact=not fresh,
                    longest_replay=longest,
                ),
                indent=2,
            )
            + "\n"
        )
        return
    import wandb

    run = wandb.init(
        entity="ocean_emulators",
        project="observational-transfer",
        name="monthly-" + a.name,
        id=hashlib.sha256(str(out).encode()).hexdigest()[:12],
        resume="allow",
        dir=str(out),
        mode="online",
        config=protocol,
    )
    if not run.url:
        raise ValueError("No online W&B URL")
    (out / "wandb-url.txt").write_text(run.url + "\n")
    print(json.dumps(dict(wandb_url=run.url, step=step)), flush=True)
    if step == (0 if fresh else 2000):
        save(out / "fork.pt")
    while step < end:
        record = update()
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
            if step in (main_end, end):
                raw = evaluate(model, data, validation)
                record["validation_raw_rmse_blend"] = selection_score(
                    raw, control, [], metric="rmse-only"
                )
                (out / f"validation-raw-{step:05d}.json").write_text(
                    json.dumps(raw, indent=2) + "\n"
                )
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
            dict(
                producer=config["producer"],
                step=step,
                counts=counts,
                best=best,
                replay_examples=replay_examples,
                fresh_examples=fresh_examples,
                source_sha256=row["sha256"],
            ),
            indent=2,
        )
        + "\n"
    )
    run.finish()


if __name__ == "__main__":
    main()
