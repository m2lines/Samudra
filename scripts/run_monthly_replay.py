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

from samudra.experiments.monthly_distributed import DistributedUpdate
from samudra.experiments.monthly_latent import MonthlyLatentOcean
from samudra.experiments.monthly_model_data import ModelMonths, model_loss, task_source
from samudra.experiments.monthly_pilot_data import observation_loss
from samudra.experiments.monthly_pilot_evaluation import evaluate as evaluate_local
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


def phase_settings(step, mode, arm, fork_step=None):
    if mode == "shared-cooldown":
        if fork_step not in (2000, 3000, 5000, 6000, 7000):
            raise ValueError("Unauthorized cooldown checkpoint")
        if not fork_step <= step < fork_step + 1000:
            raise ValueError("Update outside cooldown")
        return True, "obs", 1e-4 + (1e-6 - 1e-4) * (step - fork_step + 1) / 1000
    main = (
        8000
        if mode == "shared-8k"
        else (4000 if mode in ("extend", "fresh-4k", "shared-cool4k") else 2000)
    )
    if (
        step < (4000 if mode == "shared-cool4k" else (2000 if mode == "extend" else 0))
        or step >= main + 1000
    ):
        raise ValueError("Update outside authorized experiment")
    cooldown = step >= main
    lr = 1e-4 if not cooldown else 1e-4 + (1e-6 - 1e-4) * (step - main + 1) / 1000
    return cooldown, task_source(arm, step, cooldown), lr


def main():
    runtime = DistributedUpdate()

    def evaluate(*args, **kwargs):
        return runtime.evaluate(evaluate_local, *args, **kwargs)

    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--qualify", action="store_true")
    a = p.parse_args()
    config = json.loads((a.root / "paths.json").read_text())
    row = next(r for r in config["runs"] if r["name"] == a.name)
    replay = row["mode"] == "replay"
    shared = row["mode"] in ("shared-8k", "shared-cool4k", "shared-cooldown")
    fresh = replay or row["mode"] in ("fresh-2k", "fresh-4k", "shared-8k")
    architecture = row.get("architecture", {})
    if architecture not in (
        {},
        {"latent_width": 64, "latent_stride": 1},
        {"processor_forcing": False},
    ):
        raise ValueError("Unqualified architecture requested")
    source_path = Path(row["checkpoint"])
    source_sha256 = digest(source_path)
    if row["sha256"] is not None and source_sha256 != row["sha256"]:
        raise ValueError("Source checkpoint changed")
    if row["sha256"] is None and row["mode"] not in (
        "shared-cool4k",
        "shared-cooldown",
    ):
        raise ValueError("Only an audited same-run fork may omit its advance hash")
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
        new_main_updates=(
            8000
            if row["mode"] == "shared-8k"
            else (
                0
                if row["mode"] in ("shared-cool4k", "shared-cooldown")
                else (4000 if row["mode"] == "fresh-4k" else 2000)
            )
        ),
        cooldown=1000,
        main_end=8000
        if row["mode"] == "shared-8k"
        else (2000 if replay or row["mode"] == "fresh-2k" else 4000),
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
    if row["mode"] == "shared-cooldown":
        phase_settings(row["fork_step"], row["mode"], row["arm"], row["fork_step"])
        protocol["main_end"] = row["fork_step"]
    if shared:
        protocol.update(
            shared_era5=True,
            forcing_proof_sha256=config["forcing_proof_sha256"],
            source_sha256=source_sha256,
        )
    optimization = config.get("optimization")
    if optimization is not None:
        if (
            not shared
            or replay
            or optimization
            != dict(
                activation_checkpointing=False,
                compile_decoder=True,
                fused_adamw=True,
                microbatch=2 if runtime.world == 4 else 8,
            )
        ):
            raise ValueError("Unqualified optimization settings")
        protocol["optimization"] = optimization
    if runtime.world > 1:
        if optimization is None:
            raise ValueError("DDP requires qualified optimized batching")
        protocol["distributed"] = dict(
            world_size=4, local_examples=2, effective_batch=8
        )
    manifest = out / "protocol.json"
    if manifest.exists() and json.loads(manifest.read_text()) != protocol:
        raise ValueError("Frozen protocol changed")
    runtime.write(manifest, json.dumps(protocol, indent=2) + "\n")
    runtime.barrier()
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
    forcing_cache = None
    if shared:
        from samudra.experiments.shared_era5 import SharedEra5

        forcing_cache = SharedEra5(
            config["forcing_cache"],
            reader,
            data,
            config["forcing_proof_sha256"],
            full_readback=a.qualify,
        )
    model = MonthlyLatentOcean(**architecture).cuda()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    if optimization is not None:
        for network in (model, ema):
            network.activation_checkpointing = False
            network.decoder.activation_checkpointing = False
        model.decoder.core.forward = torch.compile(  # type: ignore[method-assign]
            model.decoder.core.forward, mode="default", fullgraph=False
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, fused=True if optimization else None
    )
    distributed_loss = runtime.wrap(model)
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
        if runtime.world > 1 and "distributed_rng" in saved:
            rank_rng = saved["distributed_rng"][runtime.rank]
            torch.set_rng_state(rank_rng["cpu"].cpu())
            torch.cuda.set_rng_state(rank_rng["cuda"].cpu())
        else:
            torch.set_rng_state(saved["cpu_rng"].cpu())
            torch.cuda.set_rng_state(saved["cuda_rng"][0].cpu())
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
            source["step"] != (protocol["main_end"] if shared else 2000)
            or source["protocol"]["arm"] != row["arm"]
            or source["protocol"]["seed"] != 1729
        ):
            raise ValueError("Source arm/step/seed differs")
        if row["mode"] in ("shared-cool4k", "shared-cooldown") and (
            source["protocol"].get("shared_era5") is not True
            or source["protocol"]["producer"]
            != row.get("source_producer", config["producer"])
            or source["protocol"]["mode"] != "shared-8k"
            or source["protocol"]["forcing_proof_sha256"]
            != config["forcing_proof_sha256"]
        ):
            raise ValueError("Cooldown fork is not this qualified shared-ERA5 run")
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
        distributed_rng = runtime.rng_states()
        if runtime.rank == 0:
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
                    cuda_rng=[torch.cuda.get_rng_state()],
                    distributed_rng=distributed_rng,
                    step=step,
                    best=best,
                    counts=counts,
                    replay_examples=replay_examples,
                    fresh_examples=fresh_examples,
                ),
                path,
            )

        runtime.barrier()

    def update():
        nonlocal step, replay_examples, fresh_examples
        if step == main_end:
            ema.load_state_dict(model.state_dict())
        cooldown, source, lr = phase_settings(
            step, row["mode"], row["arm"], row.get("fork_step")
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses, lead_days, ages, components, rms = [], [], [], [], []
        started = time.monotonic()
        sample_trace: list[str | int] = []
        draw_trace = []
        staged = []
        for micro in range(8 if runtime.rank == 0 else 0):
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
                        sample_trace.append(str(path))
                        if optimization is not None:
                            from samudra.experiments.monthly_batching import (
                                draw_digest,
                                prepare_observation,
                            )

                            item = prepare_observation(data, sample, generator)
                            staged.append(item)
                            if a.qualify:
                                draw_trace.append(draw_digest(item))
                            fresh_examples += 1
                            lead_days.append(0)
                            continue
                        if a.qualify:
                            from samudra.experiments.monthly_batching import (
                                draw_digest,
                                prepare_observation,
                            )

                            before_draw = generator.get_state()
                            probe = prepare_observation(data, sample, generator)
                            expected_draw_state = generator.get_state()
                            draw_trace.append(draw_digest(probe))
                            del probe
                            generator.set_state(before_draw)
                        loss, pieces = observation_loss(
                            model, data, sample, "surface", generator, capture=capture
                        )
                        if a.qualify and not torch.equal(
                            generator.get_state(), expected_draw_state
                        ):
                            raise ValueError("Prepared OBS draw order differs")
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
                window_index = int(rng.integers(len(reader.windows)))
                batch = reader.sample(window_index, "cuda")
                sample_trace.append(window_index)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    atmosphere = (
                        forcing_cache.load(reader, window_index, data)
                        if forcing_cache is not None
                        else None
                    )
                    if optimization is not None:
                        from samudra.experiments.monthly_batching import (
                            draw_digest,
                            prepare_model,
                        )

                        item = prepare_model(batch, generator)
                        if atmosphere is not None:
                            item["atmosphere"] = atmosphere
                        staged.append(item)
                        if a.qualify:
                            draw_trace.append(draw_digest(item))
                        continue
                    if atmosphere is not None:
                        batch["forcing"] = model.adapt(atmosphere)
                    if a.qualify:
                        from samudra.experiments.monthly_batching import (
                            draw_digest,
                            prepare_model,
                        )

                        before_draw = generator.get_state()
                        probe = prepare_model(batch, generator)
                        expected_draw_state = generator.get_state()
                        draw_trace.append(draw_digest(probe))
                        del probe
                        generator.set_state(before_draw)
                    loss = model_loss(model, batch, "surface", generator, data.area)
                    if a.qualify and not torch.equal(
                        generator.get_state(), expected_draw_state
                    ):
                        raise ValueError("Prepared model draw order differs")
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            (loss / 8).backward()
            losses.append(float(loss.detach()))
        if runtime.world > 1:
            staged = runtime.prepared(staged if runtime.rank == 0 else None)
            metadata = runtime.share(
                dict(
                    sample_trace=sample_trace,
                    draw_trace=draw_trace,
                    generator=generator.get_state(),
                    rng=rng.bit_generator.state,
                    fresh_examples=fresh_examples,
                    lead_days=lead_days,
                )
                if runtime.rank == 0
                else None
            )
            sample_trace, draw_trace = metadata["sample_trace"], metadata["draw_trace"]
            generator.set_state(metadata["generator"].cpu())
            rng.bit_generator.state = metadata["rng"]
            fresh_examples, lead_days = (
                metadata["fresh_examples"],
                metadata["lead_days"],
            )
            local_items = staged[2 * runtime.rank : 2 * runtime.rank + 2]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, local_pieces, local_rms = distributed_loss(
                    local_items, source, data.area
                )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite distributed loss")
            loss.backward()
            summaries = runtime.gather(
                dict(loss=float(loss.detach()), pieces=local_pieces, rms=local_rms)
            )
            losses = [entry["loss"] for entry in summaries]
            if source == "obs":
                components = [piece for entry in summaries for piece in entry["pieces"]]
                rms = [value for entry in summaries for value in entry["rms"]]
            del staged, local_items
        elif optimization is not None:
            from samudra.experiments.monthly_batching import batched_loss, groups

            for group in groups(staged, optimization["microbatch"]):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, pieces, latent_rms = batched_loss(
                        model, group, source, data.area
                    )
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite batched loss")
                (loss * (len(group) / 8)).backward()
                losses.extend([float(loss.detach())] * len(group))
                if source == "obs":
                    components.extend(
                        [
                            {key: float(value[index]) for key, value in pieces.items()}
                            for index in range(len(group))
                        ]
                    )
                    rms.extend(latent_rms.tolist())
            del staged
        adapter_norm = (
            sum(
                float(p.grad.detach().float().square().sum())
                for p in model.atmosphere_adapter.parameters()
                if p.grad is not None
            )
            ** 0.5
        )
        if shared and (not math.isfinite(adapter_norm) or adapter_norm == 0):
            raise ValueError(
                "Shared ERA5 adapter did not receive finite nonzero gradients"
            )
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
            adapter_gradient_norm=adapter_norm,
            obs_updates=counts["obs"],
            model_updates=counts["om4"] + counts["cm4"],
            seconds=time.monotonic() - started,
        )
        if a.qualify:
            from samudra.experiments.monthly_batching import digest_tensor

            record.update(
                sample_trace=sample_trace,
                draw_trace=draw_trace,
                generator_sha256=digest_tensor(generator.get_state()),
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
        records = []
        for qualification_index in range(config.get("qualification_updates", 4)):
            records.append(update())
            if qualification_index == 0:
                save(out / "first-update.pt")
                if runtime.rank == 0:
                    torch.save(
                        {
                            name: parameter.grad.detach().cpu()
                            for name, parameter in model.named_parameters()
                            if parameter.grad is not None
                        },
                        out / "first-gradients.pt",
                    )
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
        raw_metrics = evaluate(model, data, validation[:1])
        if not math.isfinite(selection_score(metrics, control, [], metric="rmse-only")):
            raise ValueError("Invalid diffusion evaluation")
        final_hash = hashlib.sha256()
        for name, value in model.state_dict().items():
            final_hash.update(name.encode())
            final_hash.update(value.detach().cpu().numpy().tobytes())
        rank_hashes = runtime.gather(final_hash.hexdigest())
        if len(set(rank_hashes)) != 1:
            raise ValueError("Distributed model weights differ across ranks")
        peak_cuda_bytes = max(runtime.gather(torch.cuda.max_memory_allocated()))
        runtime.write(
            out / "QUALIFIED.json",
            json.dumps(
                dict(
                    protocol=protocol,
                    exact_serialized_update=True,
                    initial_random_backbone_sha256=initial_hash,
                    records=records,
                    validation_ema=metrics,
                    validation_raw=raw_metrics,
                    replay_examples=replay_examples,
                    peak_cuda_bytes=peak_cuda_bytes,
                    rank_model_hashes=rank_hashes,
                    parameters=sum(p.numel() for p in model.parameters()),
                    latent_shape=list(model.latent_shape),
                    latent_width=model.latent_width,
                    processor_forcing_enabled=model.processor.forcing_enabled,
                    source_loading_exact=not fresh,
                    longest_replay=longest,
                ),
                indent=2,
            )
            + "\n",
        )
        return
    import wandb

    run = None
    if runtime.rank == 0:
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
            metrics = evaluate(
                ema,
                data,
                validation,
                prediction_path=(out / f"validation-predictions-{step:05d}.npz")
                if row.get("save_validation_predictions")
                else None,
            )
            value = selection_score(metrics, control, [], metric="rmse-only")
            record["validation_rmse_blend"] = value
            runtime.write(
                out / f"validation-{step:05d}.json",
                json.dumps(metrics, indent=2) + "\n",
            )
            if value < best:
                best = value
                save(out / "best.pt")
            endpoint = step in (
                (2000, 4000, main_end, end) if shared else (main_end, end)
            )
            if endpoint or row.get("raw_validation_every") == 250:
                raw = evaluate(model, data, validation)
                record["validation_raw_rmse_blend"] = selection_score(
                    raw, control, [], metric="rmse-only"
                )
                runtime.write(
                    out / f"validation-raw-{step:05d}.json",
                    json.dumps(raw, indent=2) + "\n",
                )
                if endpoint:
                    save(out / f"endpoint-{step:05d}.pt")
        # Extra continuation anchors must not add evaluation or consume RNG.
        if row["mode"] == "shared-8k" and step in (3000, 5000, 6000, 7000):
            save(out / f"endpoint-{step:05d}.pt")
        if step % 50 == 0:
            save(state_path)
        if runtime.rank == 0:
            with (out / "events.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            assert run is not None
            run.log(record, step=step)
    save(state_path)
    runtime.write(
        out / "TRAIN_COMPLETE.json",
        json.dumps(
            dict(
                producer=config["producer"],
                step=step,
                counts=counts,
                best=best,
                replay_examples=replay_examples,
                fresh_examples=fresh_examples,
                source_sha256=source_sha256,
            ),
            indent=2,
        )
        + "\n",
    )
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
