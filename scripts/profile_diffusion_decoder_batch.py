# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Bounded decoder forward/backward batching profile; never updates model weights.

The latent and wet mask come from a real observation and a fixed checkpoint.
The noisy physical input and zero regression target are synthetic: these timings
measure a single decoder call, not a sampler, scientific loss, or optimizer step.
Identical examples are repeated to compare mean-loss gradients against serial
batch-one execution. JSON is flushed after every completed configuration.
"""

import argparse
import gc
import os
import statistics
import time
from pathlib import Path

import torch

from samudra.experiments.diffusion_direct import load_latent_checkpoint
from samudra.experiments.diffusion_global import make_model
from samudra.experiments.diffusion_noise import diffusion_noise
from samudra.experiments.observation_pilot import atomic_json, digest
from samudra.experiments.observation_training import Samples


def snapshot(decoder, output, latent, batch):
    """Retain only CPU copies outside the measured interval."""
    return {
        "output": output.detach()[0].float().cpu(),
        "latent_gradient": (latent.grad.detach()[0].float() * batch).cpu(),
        "parameter_gradients": {
            name: p.grad.detach().float().cpu().clone()
            for name, p in decoder.named_parameters()
            if p.grad is not None
        },
    }


def relative_error(value, reference):
    return float(
        (value.double() - reference.double()).square().sum().sqrt()
        / reference.double().square().sum().sqrt().clamp_min(1e-30)
    )


def differences(value, reference):
    gradients = value["parameter_gradients"]
    original = reference["parameter_gradients"]
    numerator, denominator = 0.0, 0.0
    for name, grad in gradients.items():
        numerator += float((grad.double() - original[name].double()).square().sum())
        denominator += float(original[name].double().square().sum())
    return dict(
        output_relative_l2=relative_error(value["output"], reference["output"]),
        output_max_absolute=float((value["output"] - reference["output"]).abs().max()),
        latent_gradient_relative_l2=relative_error(
            value["latent_gradient"], reference["latent_gradient"]
        ),
        parameter_gradient_relative_l2=(numerator / max(denominator, 1e-30)) ** 0.5,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-seconds", type=float, default=720)
    parser.add_argument("--max-memory-fraction", type=float, default=0.90)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    args = parser.parse_args()
    if min(args.batches) < 1 or args.repeats < 1 or args.warmup < 1:
        raise ValueError("Positive batches, repeats and warmup required")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(1701)
    torch.cuda.set_per_process_memory_fraction(args.max_memory_fraction)
    started = time.monotonic()
    checkpoint = args.checkpoint or args.root / "checkpoints/parent-16256.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    data = Samples(
        args.root / "data/observations",
        "cuda",
        surface_fill="zero",
        global_observations=True,
    )
    data.use_observation_normalization()
    for key, filename in (
        ("observation_manifest", "SHA256SUMS"),
        ("stats", "statistics.npz"),
        ("grid", "grid.npz"),
    ):
        if digest(data.root / filename) != saved["contract"][key]:
            raise ValueError(f"Checkpoint/data mismatch: {key}")
    provenance = dict(
        checkpoint=str(checkpoint),
        checkpoint_sha256=digest(checkpoint),
        counts=saved["counts"],
        step=saved["step"],
    )
    model = make_model(data.grid["names"].tolist(), "cuda").eval()
    load_latent_checkpoint(model, saved)
    del saved
    source = data.paths("train")[0]
    sample = data.load(source)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        atmosphere = model.adapt(sample["atmosphere"][:, :19])
        latent, _, _ = model.encode_native(
            sample["surface"][:, :19],
            atmosphere,
            sample["contexts"][:, 18],
            data.mask,
            sample["validity"][:, :19],
            task="observation",
        )
    base_latent = latent.flatten(1, 2).detach().float()
    mask = data.mask.repeat(2, 1, 1)
    generator = torch.Generator(device="cuda").manual_seed(1702)
    noisy = diffusion_noise(
        (1, model.decoder.fields, *mask.shape[-2:]),
        mask,
        generator,
        model.decoder.noise_correlation,
        False,
    )
    decoder = model.decoder.train()
    del model, sample, data, latent, atmosphere
    gc.collect()
    torch.cuda.empty_cache()
    if args.channels_last:
        decoder.to(memory_format=torch.channels_last)
    results = dict(
        scope="single decoder forward+backward; no sampler, optimizer, DDP or full-update claim",
        data="real observation latent and mask; synthetic correlated noisy input and zero target",
        source=str(source),
        source_sha256=digest(source),
        provenance=provenance,
        device=torch.cuda.get_device_name(),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        job_id=os.environ.get("SLURM_JOB_ID"),
        parameter_dtype=str(next(decoder.parameters()).dtype),
        autocast="disabled" if args.fp32 else "bfloat16",
        compiled=not args.eager,
        compile_mode="default",
        channels_last=args.channels_last,
        latent_shape=list(base_latent.shape),
        noisy_shape=list(noisy.shape),
        objective="mean of per-example mean-square masked output at sigma=1",
        identical_repeated_examples=True,
        warmup=args.warmup,
        repeats=args.repeats,
        records=[],
        status="running",
    )
    atomic_json(results, args.output)

    def step(batch):
        decoder.zero_grad(set_to_none=True)
        z = base_latent.repeat(batch, 1, 1, 1).detach()
        x = noisy.repeat(batch, 1, 1, 1)
        if args.channels_last:
            z = z.contiguous(memory_format=torch.channels_last)
            x = x.contiguous(memory_format=torch.channels_last)
        z.requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not args.fp32):
            prediction = decoder(x, torch.ones(batch, device="cuda"), z, mask)
            objective = prediction.float().square().flatten(1).mean(1).mean()
        objective.backward()
        return prediction, z, objective

    # Serial eager execution defines the equivalence reference; identical batch
    # examples make this equivalent to serial mean-loss gradient accumulation.
    output, z, objective = step(1)
    reference = snapshot(decoder, output, z, 1)
    results["serial_reference_loss"] = float(objective.detach())
    del output, z, objective
    decoder.zero_grad(set_to_none=True)
    if not args.eager:
        decoder.compile()
    for batch in args.batches:
        if time.monotonic() - started >= args.max_seconds:
            results["status"] = "time_budget_exhausted"
            break
        record = dict(batch=batch, status="running")
        results["records"].append(record)
        try:
            torch.cuda.reset_peak_memory_stats()
            warm_started = time.monotonic()
            for _ in range(args.warmup):
                output, z, objective = step(batch)
                del output, z, objective
            torch.cuda.synchronize()
            record["warmup_wall_seconds"] = time.monotonic() - warm_started
            cuda_times, wall_times = [], []
            for index in range(args.repeats):
                torch.cuda.synchronize()
                before, after = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                wall_started = time.monotonic()
                before.record()
                output, z, objective = step(batch)
                after.record()
                torch.cuda.synchronize()
                wall_times.append(time.monotonic() - wall_started)
                cuda_times.append(before.elapsed_time(after) / 1000)
                if index == args.repeats - 1:
                    record["loss"] = float(objective.detach())
                    record["equivalence_vs_serial_eager"] = differences(
                        snapshot(decoder, output, z, batch), reference
                    )
                del output, z, objective
            record.update(
                status="complete",
                cuda_seconds=cuda_times,
                wall_seconds=wall_times,
                median_wall_seconds=statistics.median(wall_times),
                examples_per_second=batch / statistics.median(wall_times),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            )
        except torch.OutOfMemoryError as error:
            record.update(status="out_of_memory", error=str(error))
            results["status"] = "stopped_after_oom"
            break
        finally:
            decoder.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            results["elapsed_seconds"] = time.monotonic() - started
            atomic_json(results, args.output)
        print(record, flush=True)
    else:
        results["status"] = "complete"
    results["elapsed_seconds"] = time.monotonic() - started
    atomic_json(results, args.output)


if __name__ == "__main__":
    main()
