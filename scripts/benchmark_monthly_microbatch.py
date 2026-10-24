#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Real-example microbatch screening with fixed effective batch and per-item targets."""

import argparse
import gc
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from samudra.experiments import monthly_latent
from samudra.experiments.diffusion_noise import diffusion_noise
from samudra.experiments.monthly_latent import (
    MonthlyLatentOcean,
    initializer_visibility,
)
from samudra.experiments.monthly_model_data import ModelMonths
from samudra.experiments.monthly_pilot_data import observation_batch
from samudra.experiments.observation_training import Samples


def checksum(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def records(data, reader, source):
    """Eight distinct months with common trajectory length, individual month weights."""
    result: list[dict[str, Any]] = []
    length = None
    generator = torch.Generator(device="cuda").manual_seed(1729)
    candidates = data.paths("train") if source == "obs" else range(len(reader.windows))
    for candidate in candidates:
        if source == "obs":
            sample = data.load(candidate)
            weights = sample["month_weights"]
            if length is not None and len(weights) != length:
                continue
            batch = observation_batch(data, sample, "surface", generator)
            lead = int(
                torch.randint(
                    len(weights),
                    (),
                    generator=torch.Generator().manual_seed(1729 + len(result)),
                )
            )
            target = torch.zeros_like(batch["target"])
            support = torch.zeros_like(batch["support"])
            target[:, [38, 76]] = sample["surface"][:, 19 + lead]
            support[:, [38, 76]] = sample["validity"][:, 19 + lead].bool()
            initial = torch.zeros_like(target)
            support0 = torch.zeros_like(support)
            initial[:, [38, 76]] = sample["surface"][:, 18]
            support0[:, [38, 76]] = sample["validity"][:, 18].bool()
            item = dict(
                fields=batch["fields"],
                available=batch["available"],
                atmosphere=sample["atmosphere"],
                contexts=sample["contexts"],
                targets=[batch["target"], target, initial],
                supports=[batch["support"], support, support0],
                weights=weights,
                duration=float(pd.Timestamp(sample["name"]).days_in_month),
                identity=str(candidate),
                lead=lead,
            )
        else:
            window = reader.windows[candidate]
            if length is not None and len(window[2]) != length:
                continue
            batch = reader.sample(candidate, "cuda")
            weights = batch["weights"]
            lead = int(
                torch.randint(
                    len(weights),
                    (),
                    generator=torch.Generator().manual_seed(1729 + len(result)),
                )
            )
            future = batch["fields"][:, 19:]
            mask = batch["mask"][None]
            item = dict(
                fields=batch["fields"],
                available=initializer_visibility(batch["mask"], "surface", generator)[
                    None, None
                ].expand_as(batch["fields"]),
                forcing=batch["forcing"],
                contexts=batch["contexts"],
                targets=[
                    (future * weights[None, :, None, None, None]).sum(1),
                    future[:, lead],
                    batch["fields"][:, 18],
                ],
                supports=[mask] * 3,
                weights=weights,
                duration=float(batch["duration"]),
                identity=batch["first"],
                lead=lead,
            )
        length = len(weights)
        item["index"] = len(result)
        result.append(item)
        if len(result) == 8:
            return result
    raise ValueError("Fewer than eight compatible real examples")


def decoder_loss(model, latent, items, task, area, update):
    """Same masked denoising formula; independent matching RNG for each example."""
    targets = torch.cat([item["targets"][task] for item in items])
    support = torch.cat(
        [item["supports"][task] for item in items]
    ).bool() & torch.isfinite(targets)
    clean = torch.where(support, targets, 0).float()
    sigmas, noises, conditions = [], [], []
    for index, item in enumerate(items):
        generator = torch.Generator(device=latent.device).manual_seed(
            1729 + 100000 * update + 100 * item["index"] + task
        )
        sigma = (
            torch.randn(1, device=latent.device, generator=generator) * 1.2 - 1.2
        ).exp()
        mask = support[index : index + 1]
        noise = diffusion_noise(
            mask.shape, mask, generator, model.decoder.core.noise_correlation
        )
        duration = item["duration"] if task == 0 else 5.0
        position = (
            duration / 2
            if task == 0
            else (2.5 + 5 * item["lead"] if task == 1 else -2.5)
        )
        conditions.append(
            model.decoder.condition(latent[index : index + 1], mask, duration, position)
        )
        sigmas.append(sigma)
        noises.append(noise)
    sigma = torch.cat(sigmas)
    noisy = (clean + sigma[:, None, None, None] * torch.cat(noises)) * support
    prediction = model.decoder.core(noisy, sigma, torch.cat(conditions), support)
    weights = support * area
    denominator = weights.sum((-2, -1))
    present = denominator > 0
    errors = ((prediction - clean).square() * weights).sum(
        (-2, -1)
    ) / denominator.clamp_min(1e-12)
    per_item = (errors * present).sum(-1) / present.sum(-1)
    return (per_item * (1 + sigma.square()) / sigma.square()).mean()


def objective(model, items, source, area, update):
    fields = torch.cat([x["fields"] for x in items])
    availability = torch.cat([x["available"] for x in items])
    forcing = (
        model.adapt(torch.cat([x["atmosphere"] for x in items]))
        if source == "obs"
        else torch.cat([x["forcing"] for x in items])
    )
    states = model.trajectory(
        fields, availability, forcing, torch.cat([x["contexts"] for x in items])
    )
    monthly = sum(
        state * torch.stack([x["weights"][j] for x in items])[:, None, None, None, None]
        for j, state in enumerate(states[1:])
    )
    native = torch.cat([states[x["lead"] + 1][i : i + 1] for i, x in enumerate(items)])
    loss_weights = (0.8, 0.2, 0.1) if source == "obs" else (0.5, 0.5, 0.1)
    return sum(
        weight * decoder_loss(model, latent, items, task, area, update)
        for task, (weight, latent) in enumerate(
            zip(loss_weights, (monthly, native, states[0]), strict=True)
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--microbatch", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--fused", action="store_true")
    args = parser.parse_args()
    name = (
        f"micro{args.microbatch}"
        + ("-compile" if args.compile else "")
        + ("-fused" if args.fused else "")
    )
    out = args.root / name
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    legacy = json.loads(Path(config["legacy_qualification"]).read_text())
    for path, sha in legacy["data_contracts"].items():
        if checksum(path) != sha:
            raise ValueError("Changed data contract: " + path)
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    # Scoped to this diagnostic process only, including initializer checkpoint.
    monthly_latent.checkpoint = lambda function, *inputs, **kwargs: function(*inputs)
    data = Samples(
        config["samples"],
        "cuda",
        surface_fill="zero",
        global_observations=True,
        cache_size=8,
    )
    data.cache_prepared = True
    reader = ModelMonths(
        Path(config["om4"]) / "OM4.zarr",
        "om4",
        data.grid,
        config["om4"],
        packed=legacy["packed_om4"],
    )
    result = dict(
        microbatch=args.microbatch,
        effective_batch=8,
        accumulation=8 // args.microbatch,
        compile=args.compile,
        fused=args.fused,
        seed=1729,
        torch_version=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        sources={},
    )
    for source in ("om4", "obs"):
        started = time.monotonic()
        items = records(data, reader, source)
        load_seconds = time.monotonic() - started
        torch.manual_seed(1729)
        torch.cuda.manual_seed_all(1729)
        model = MonthlyLatentOcean().cuda().train()
        if args.compile:
            model.decoder.core.forward = torch.compile(  # type: ignore[method-assign]
                model.decoder.core.forward, mode="default", fullgraph=False
            )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-4, fused=True if args.fused else None
        )
        initial = torch.cat([p.detach().cpu().flatten() for p in model.parameters()])
        torch.cuda.reset_peak_memory_stats()
        timings = []
        agreement = {}
        for update in range(5):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            start = time.monotonic()
            losses = []
            for low in range(0, 8, args.microbatch):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = objective(
                        model,
                        items[low : low + args.microbatch],
                        source,
                        data.area,
                        update,
                    )
                (loss / (8 // args.microbatch)).backward()
                losses.append(float(loss.detach()))
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            torch.cuda.synchronize()
            timings.append(
                dict(
                    seconds=time.monotonic() - start,
                    loss=sum(losses) / len(losses),
                    gradient_norm=float(norm),
                )
            )
            if update == 0:
                gradients = torch.cat(
                    [
                        p.grad.detach().cpu().flatten()
                        if p.grad is not None
                        else torch.zeros(p.numel())
                        for p in model.parameters()
                    ]
                )
                after = torch.cat(
                    [p.detach().cpu().flatten() for p in model.parameters()]
                )
                reference = (
                    args.root
                    / (
                        "micro1"
                        + ("-compile" if args.compile else "")
                        + ("-fused" if args.fused else "")
                    )
                    / f"{source}-reference.pt"
                )
                if args.microbatch == 1:
                    torch.save(
                        dict(
                            gradients=gradients,
                            parameters=after,
                            initial=initial,
                            loss=timings[0]["loss"],
                        ),
                        reference,
                    )
                else:
                    baseline = torch.load(reference, weights_only=True)
                    if not torch.equal(initial, baseline["initial"]):
                        raise ValueError("Initialization changed")
                    agreement = dict(
                        loss_relative=abs(timings[0]["loss"] - baseline["loss"])
                        / abs(baseline["loss"]),
                        gradient_relative_l2=float(
                            (gradients - baseline["gradients"]).norm()
                            / baseline["gradients"].norm()
                        ),
                        update_relative_l2=float(
                            (after - baseline["parameters"]).norm()
                            / (baseline["parameters"] - initial).norm()
                        ),
                        max_parameter_difference=float(
                            (after - baseline["parameters"]).abs().max()
                        ),
                    )
                    del baseline
                del gradients, after
        median = statistics.median(x["seconds"] for x in timings[2:])
        result["sources"][source] = dict(
            examples=[x["identity"] for x in items],
            load_seconds=load_seconds,
            timings=timings,
            median_seconds=median,
            examples_per_second=8 / median,
            agreement=agreement,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            device_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        )
        (out / "RESULTS.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result["sources"][source]), flush=True)
        del model, optimizer, items, initial
        gc.collect()
        torch.cuda.empty_cache()
    from torch._dynamo.utils import counters

    result["compile_counters"] = {k: dict(v) for k, v in counters.items()}
    (out / "COMPLETE.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
