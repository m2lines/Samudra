# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Isolated full-grid profiles; repeated cached examples isolate compute from I/O."""

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

VARIANTS = {
    "baseline": (1, True, False, False),
    "batch2": (2, True, False, False),
    "batch4": (4, True, False, False),
    "batch8": (8, True, False, False),
    "no-checkpoint": (1, False, False, False),
    "compile": (1, True, True, False),
    "fused": (1, True, False, True),
    "batch2-no-checkpoint": (2, False, False, False),
}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=list(VARIANTS))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=1200)
    return parser.parse_args()


def fixed_denoiser_batch_parity(model, native):
    """Same physical tensors and sigma: compare batched versus serial gradients."""
    fields = native["fields"]
    condition = torch.cat(
        (
            fields[:, 17:19].flatten(1, 2),
            native["atmosphere"][:, 19],
            native["contexts"][:, 19],
        ),
        1,
    )
    count = 2
    generator = torch.Generator(device="cuda").manual_seed(912)
    noisy = fields[:, 19].expand(count, -1, -1, -1).clone()
    noisy += torch.randn(noisy.shape, device=noisy.device, generator=generator)
    sigma = noisy.new_tensor([0.5, 1.0])
    condition = condition.expand(count, -1, -1, -1)
    results = []
    for batch in (1, 2):
        model.zero_grad(set_to_none=True)
        outputs = []
        for start in range(0, count, batch):
            predicted = model.processor(
                noisy[start : start + batch],
                sigma[start : start + batch],
                condition[start : start + batch],
                native["mask"],
            )
            (predicted.square().mean() * batch / count).backward()
            outputs.append(predicted.detach())
        gradient = torch.cat(
            [
                p.grad.detach().float().flatten().cpu()
                for p in model.processor.parameters()
                if p.grad is not None
            ]
        )
        results.append((torch.cat(outputs).cpu(), gradient))
    output_error = float((results[0][0] - results[1][0]).abs().max())
    gradient_error = float(
        (results[0][1] - results[1][1]).norm() / results[0][1].norm().clamp_min(1e-12)
    )
    return dict(
        precision="fp32",
        examples=count,
        output_max_abs=output_error,
        gradient_relative_l2=gradient_error,
    )


def run_variant(args):
    from samudra.experiments.monthly_model_data import ModelMonths
    from samudra.experiments.observation_training import Samples
    from samudra.experiments.physical_state_batching import collate_examples
    from samudra.experiments.physical_state_objectives import (
        native_objective,
        observation_batch_objective,
    )
    from samudra.experiments.physical_state_train import make_model
    from samudra.experiments.shared_era5 import SharedEra5

    micro, checkpointing, compile_core, fused = VARIANTS[args.variant]
    torch.set_num_threads(1)
    config = json.loads(args.reference.read_text())
    legacy = json.loads(Path(config["legacy_qualification"]).read_text())
    data = Samples(
        config["samples"], "cuda", surface_fill="zero", global_observations=True
    )
    reader = ModelMonths(
        Path(config["om4"]) / "OM4.zarr",
        "om4",
        data.grid,
        config["om4"],
        packed=legacy["packed_om4"],
    )
    cache = SharedEra5(
        config["forcing_cache"],
        reader,
        data,
        config["forcing_proof_sha256"],
        full_readback=True,
    )
    obs = data.load(data.paths("train")[0])
    native = reader.sample(0, "cuda")
    native["atmosphere"] = cache.load(reader, 0, data)
    contract = dict(widths=[64, 128, 256], sampling_steps=8, noise_correlation=0.5)
    torch.manual_seed(1729)
    model = make_model(data.grid["names"].tolist(), "cuda", contract)
    model.train()
    initial_weights = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }

    def settings(checkpoint, compiled):
        model.checkpoint_denoiser = checkpoint
        if compiled:
            model.enable_compile()
        else:
            for diffuser in (model.initializer, model.processor):
                if hasattr(diffuser, "_compiled_net"):
                    del diffuser._compiled_net

    def update(task, optimizer=None):
        model.zero_grad(set_to_none=True)
        value = 0.0
        for start in range(0, 8, micro):
            generator = torch.Generator(device="cuda").manual_seed(24000 + start)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = (
                    observation_batch_objective(
                        model, data, [obs] * micro, generator, members=2
                    )
                    if task == "observation"
                    else native_objective(
                        model, data, collate_examples([native] * micro), generator
                    )
                )
            (loss * micro / 8).backward()
            value += float(loss.detach()) * micro / 8
        if optimizer is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
        return value

    report = dict(
        variant=args.variant,
        microbatch=micro,
        effective_batch=8,
        checkpointing=checkpointing,
        compiled=compile_core,
        fused=fused,
        samples="same first training month repeated eight times; no I/O timed",
        noise_note="same noise within parity test; microbatch variants assign different noise",
        device=torch.cuda.get_device_name(),
        tasks={},
    )
    if args.variant == "baseline":
        report["fixed_denoiser_batch_parity"] = fixed_denoiser_batch_parity(
            model, native
        )
    for task in ("om4", "observation"):
        model.load_state_dict(initial_weights)
        # Compare identical microbatch shape/draws before any optimizer step.
        settings(True, False)
        reference_loss = update(task)
        reference_grad = torch.cat(
            [
                p.grad.detach().float().flatten().cpu()
                for p in model.parameters()
                if p.grad is not None
            ]
        )
        settings(checkpointing, compile_core)
        torch.cuda.synchronize()
        started = time.monotonic()
        candidate_loss = update(task)
        torch.cuda.synchronize()
        cold = time.monotonic() - started
        candidate_grad = torch.cat(
            [
                p.grad.detach().float().flatten().cpu()
                for p in model.parameters()
                if p.grad is not None
            ]
        )
        relative = float(
            (candidate_grad - reference_grad).norm()
            / reference_grad.norm().clamp_min(1e-12)
        )
        cosine = float(
            torch.nn.functional.cosine_similarity(reference_grad, candidate_grad, dim=0)
        )
        if not torch.isfinite(candidate_grad).all():
            raise FloatingPointError("Nonfinite profile gradients")
        del reference_grad, candidate_grad
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-4, weight_decay=0.01, fused=fused
        )
        times, peaks = [], []
        for _ in range(args.repeats):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.monotonic()
            loss = update(task, optimizer)
            torch.cuda.synchronize()
            times.append(time.monotonic() - started)
            peaks.append(torch.cuda.max_memory_allocated() / 2**30)
        report["tasks"][task] = dict(
            cold_seconds=cold,
            warm_seconds=times,
            peak_gib=peaks,
            loss=loss,
            reference_loss=reference_loss,
            candidate_loss=candidate_loss,
            gradient_relative_l2=relative,
            gradient_cosine=cosine,
        )
        del optimizer
        gc.collect()
    report["status"] = "completed"
    return report


def main():
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.variant:
        path = args.output / f"{args.variant}.json"
        try:
            result = run_variant(args)
        except torch.cuda.OutOfMemoryError as exc:
            result = dict(variant=args.variant, status="oom", error=str(exc))
        path.write_text(json.dumps(result, indent=2))
        return
    results = []
    for name in VARIANTS:
        command = [
            sys.executable,
            __file__,
            "--reference",
            str(args.reference),
            "--output",
            str(args.output),
            "--variant",
            name,
            "--repeats",
            str(args.repeats),
        ]
        with (args.output / f"{name}.log").open("w") as stream:
            try:
                process = subprocess.run(
                    command,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                    check=False,
                )
                status = "exited" if process.returncode == 0 else "failed"
            except subprocess.TimeoutExpired:
                status = "timeout"
        path = args.output / f"{name}.json"
        results.append(
            json.loads(path.read_text())
            if path.exists()
            else dict(variant=name, status=status)
        )
        (args.output / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
