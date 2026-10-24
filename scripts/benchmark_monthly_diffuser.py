#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Disposable real-batch performance experiments; never writes training checkpoints."""

import argparse
import gc
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from samudra.experiments import monthly_latent
from samudra.experiments.joint_diffusion import JointInteriorDecoder
from samudra.experiments.monthly_latent import MonthlyLatentOcean
from samudra.experiments.monthly_model_data import ModelMonths, model_loss
from samudra.experiments.monthly_pilot_data import observation_loss
from samudra.experiments.observation_training import Samples

VARIANTS = (
    "baseline",
    "decoder-no-checkpoint",
    "all-no-checkpoint",
    "compile-decoder",
    "compile-decoder-no-checkpoint",
    "initializer-no-checkpoint",
    "fused-adamw",
)


def checksum(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--measured", type=int, default=3)
    args = parser.parse_args()
    out = args.root / args.variant
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    legacy = json.loads(Path(config["legacy_qualification"]).read_text())
    for path, sha in legacy["data_contracts"].items():
        if checksum(path) != sha:
            raise ValueError("Data contract changed: " + path)
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
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
    sample_path = data.paths("train")[0]
    loads = {}
    batches = {}
    for source, load in (
        ("obs", lambda: data.load(sample_path)),
        ("om4", lambda: reader.sample(0, "cuda")),
    ):
        times = []
        for _ in range(2):
            torch.cuda.synchronize()
            started = time.monotonic()
            batches[source] = load()
            torch.cuda.synchronize()
            times.append(time.monotonic() - started)
        loads[source] = dict(cold_seconds=times[0], repeat_seconds=times[1])
    original_checkpoint = monthly_latent.checkpoint

    def diagnostic_checkpoint(function, *inputs, **kwargs):
        decoder = isinstance(function, JointInteriorDecoder)
        # compile wraps a bound forward, not the module: isinstance stays valid.
        skip = (
            args.variant == "all-no-checkpoint"
            or (
                decoder
                and args.variant
                in ("decoder-no-checkpoint", "compile-decoder-no-checkpoint")
            )
            or (not decoder and args.variant == "initializer-no-checkpoint")
        )
        return (
            function(*inputs)
            if skip
            else original_checkpoint(function, *inputs, **kwargs)
        )

    monthly_latent.checkpoint = diagnostic_checkpoint
    results = dict(
        variant=args.variant,
        data_load=loads,
        observation=sample_path.name,
        model_window=0,
        torch_version=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        effective_batch=8,
        measured_updates=args.measured,
        dtype="bfloat16",
        losses="unchanged monthly+forecast+reconstruction",
        sources={},
    )
    for source in ("om4", "obs"):
        torch.manual_seed(1729)
        torch.cuda.manual_seed_all(1729)
        model = MonthlyLatentOcean().cuda().train()
        if args.variant.startswith("compile-"):
            model.decoder.core.forward = torch.compile(  # type: ignore[method-assign]
                model.decoder.core.forward, mode="default", fullgraph=True
            )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            fused=True if args.variant == "fused-adamw" else None,
        )
        generator = torch.Generator(device="cuda").manual_seed(1729)
        initial = torch.cat([p.detach().flatten().cpu() for p in model.parameters()])

        def update(model=model, optimizer=optimizer, generator=generator):
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            torch.cuda.synchronize()
            started = time.monotonic()
            for _ in range(8):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if source == "om4":
                        loss = model_loss(
                            model, batches[source], "surface", generator, data.area
                        )
                    else:
                        loss, _ = observation_loss(
                            model, data, batches[source], "surface", generator
                        )
                (loss / 8).backward()
                total += float(loss.detach()) / 8
            torch.cuda.synchronize()
            backward_finished = time.monotonic()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            torch.cuda.synchronize()
            return dict(
                loss=total,
                gradient_norm=float(norm),
                forward_backward_seconds=backward_finished - started,
                update_seconds=time.monotonic() - started,
            )

        torch.cuda.reset_peak_memory_stats()
        cold = update()
        gradients = torch.cat(
            [
                (
                    p.grad.detach().flatten().cpu()
                    if p.grad is not None
                    else torch.zeros(p.numel())
                )
                for p in model.parameters()
            ]
        )
        after = torch.cat([p.detach().flatten().cpu() for p in model.parameters()])
        reference_path = args.root / "baseline" / f"{source}-reference.pt"
        if args.variant == "baseline":
            torch.save(
                dict(
                    loss=cold["loss"],
                    gradients=gradients,
                    parameters=after,
                    initial=initial,
                ),
                reference_path,
            )
            agreement: dict[str, object] = dict(reference=True)
        else:
            reference = torch.load(reference_path, weights_only=True)
            if not torch.equal(initial, reference["initial"]):
                raise ValueError("Initialization differs")
            grad_relative = float(
                (gradients - reference["gradients"]).norm()
                / reference["gradients"].norm().clamp_min(1e-20)
            )
            update_relative = float(
                (after - reference["parameters"]).norm()
                / (reference["parameters"] - initial).norm().clamp_min(1e-20)
            )
            agreement = dict(
                loss_relative=abs(cold["loss"] - reference["loss"])
                / abs(reference["loss"]),
                gradient_relative_l2=grad_relative,
                update_relative_l2=update_relative,
                max_parameter_difference=float(
                    (after - reference["parameters"]).abs().max()
                ),
                approximate_gate=grad_relative < 0.02 and update_relative < 0.05,
            )
            del reference
        del gradients, after, initial
        warmup = update()
        timed = [update() for _ in range(args.measured)]
        results["sources"][source] = dict(
            cold_update=cold,
            warmup=warmup,
            measured=timed,
            median_update_seconds=statistics.median(x["update_seconds"] for x in timed),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            agreement=agreement,
        )
        (out / "RESULTS.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(dict(source=source, **results["sources"][source])), flush=True)
        del model, optimizer, generator
        gc.collect()
        torch.cuda.empty_cache()
    from torch._dynamo.utils import counters

    results["compile_counters"] = {key: dict(value) for key, value in counters.items()}
    (out / "COMPLETE.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
