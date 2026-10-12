# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Matched current full-sampler observation profiling; weights never updated."""

import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

import samudra.experiments.diffusion_latent_forecast as forecast_module
from samudra.experiments.diffusion_global import make_model, observation_parts
from samudra.experiments.diffusion_interventions import DIRECT_PARENT_SHA
from samudra.experiments.diffusion_noise import diffusion_noise
from samudra.experiments.observation_pilot import atomic_json, digest
from samudra.experiments.observation_training import Samples
from samudra.experiments.task_schedule import sample_indices


def sample_joint_no_sync(
    decoder,
    latent,
    mask,
    generator,
    steps=16,
    *,
    known_values=None,
    known_mask=None,
    checkpoint_denoiser=False,
):
    """Same Heun arithmetic; last-interval branch uses its known Python index."""
    if steps < 2 or (known_values is None) != (known_mask is None):
        raise ValueError("Invalid sampler arguments")
    shape = (latent.shape[0], decoder.fields, *mask.shape[-2:])
    if mask.shape[-3] != decoder.fields:
        raise ValueError("Target mask does not match decoder fields")

    def anchor(value):
        if known_mask is not None:
            value = torch.where(known_mask.bool(), known_values, value)
        return value * mask

    def denoise(value, sigma):
        sigma_batch = sigma.expand(latent.shape[0])
        if checkpoint_denoiser and torch.is_grad_enabled():
            result = checkpoint(
                decoder, value, sigma_batch, latent, mask, use_reentrant=False
            )
        else:
            result = decoder(value, sigma_batch, latent, mask)
        return anchor(result.float())

    sigmas = (
        80 ** (1 / 7)
        + torch.linspace(0, 1, steps, device=latent.device)
        * (0.002 ** (1 / 7) - 80 ** (1 / 7))
    ) ** 7
    sigmas = torch.cat((sigmas, sigmas.new_zeros(1)))
    state = anchor(
        diffusion_noise(
            shape,
            mask,
            generator,
            decoder.noise_correlation,
            decoder.paired_noise_draws,
        )
        * sigmas[0]
    )
    for index, (current, following) in enumerate(
        zip(sigmas[:-1], sigmas[1:], strict=True)
    ):
        derivative = (state - denoise(state, current)) / current
        trial = anchor(state + (following - current) * derivative)
        if index < steps - 1:
            next_derivative = (trial - denoise(trial, following)) / following
            trial = anchor(
                state + (following - current) * (derivative + next_derivative) / 2
            )
        state = trial
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["base", "no_sampler_sync", "channels_last"],
        choices=["base", "no_sampler_sync", "channels_last"],
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world not in (1, 2, 4, 8):
        raise ValueError("World size must divide effective batch eight")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.set_num_threads(1)
    torch.manual_seed(271829)
    if world > 1:
        dist.init_process_group("nccl", device_id=device)
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    if world > 1:
        dist.barrier()
    torch._dynamo.config.recompile_limit = 64
    torch._dynamo.config.accumulated_recompile_limit = 256
    path = args.root / "checkpoints/parent-16256.pt"
    if digest(path) != DIRECT_PARENT_SHA:
        raise ValueError("Unexpected parent checkpoint")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved["step"] != 16256 or saved["counts"] != {"om4": 8128, "observation": 8128}:
        raise ValueError("Unexpected parent counts")
    data = Samples(
        args.root / "data/observations",
        device,
        surface_fill="zero",
        global_observations=True,
    )
    data.use_observation_normalization()
    for key, source in (
        ("observation_manifest", data.root / "SHA256SUMS"),
        ("stats", data.root / "statistics.npz"),
        ("grid", data.root / "grid.npz"),
        ("reference", args.root / "checkpoints/selection-reference.json"),
    ):
        if digest(source) != saved["contract"][key]:
            raise ValueError(f"Changed input: {key}")
    paths = data.paths("train")
    if len(paths) != 243 or not bool((data.area > 0).all()):
        raise ValueError("Changed cohort or latitude mask")
    indices = sample_indices(len(paths), 271829, 0, 8)
    local_micro = list(range(rank, 8, world))
    before = time.perf_counter()
    samples = {micro: data.load(paths[indices[micro]]) for micro in local_micro}
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - before
    result = dict(
        scope="Full observation forecast, completion and monthly reconstruction losses, backward, gradient reduction, norm and clipping; fixed weights, no optimizer updates. Effective batch eight, two members, 32 Heun steps. Excludes OM4, checkpoint saves and queueing. Manual gradient sum across ranks preserves serial /8 scaling.",
        gpu=torch.cuda.get_device_name(),
        world_size=world,
        rank=rank,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(),
        bf16_autocast=True,
        compile_decoder=True,
        parent_sha256=DIRECT_PARENT_SHA,
        producer=os.environ.get("SAMUDRA_CODE_COMMIT"),
        job=os.environ.get("SLURM_JOB_ID"),
        contract=saved["contract"],
        sample_indices=indices,
        samples=[
            dict(
                micro=m,
                path=str(paths[indices[m]]),
                sha256=digest(paths[indices[m]]),
                seed=2271829 + m,
            )
            for m in range(8)
        ],
        local_micro=local_micro,
        load_seconds=load_seconds,
        variants={},
    )
    original_sampler = forecast_module.sample_joint
    for variant in args.variants:
        torch._dynamo.utils.counters.clear()
        forecast_module.sample_joint = (
            sample_joint_no_sync if variant == "no_sampler_sync" else original_sampler
        )
        model = make_model(data.grid["names"].tolist(), device)
        model.load_state_dict(saved["model"], strict=True)
        model.train()
        if variant == "channels_last":
            model.to(memory_format=torch.channels_last)
        model.decoder.compile()
        records = []

        def update(micros, model, distributed=True):
            model.zero_grad(set_to_none=True)
            if world > 1 and distributed:
                dist.barrier()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            total = torch.zeros((), device=device)
            objectives: dict[str, float] = {}
            stage_seconds: dict[str, float] = {}
            for micro in micros:
                stage_start = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    for label, loss, measurements in observation_parts(
                        model, data, samples[micro], 2271829 + micro
                    ):
                        if not bool(torch.isfinite(loss)):
                            raise FloatingPointError(label)
                        (loss / 8).backward()
                        torch.cuda.synchronize()
                        stage_end = time.perf_counter()
                        stage_seconds[label] = (
                            stage_seconds.get(label, 0.0) + stage_end - stage_start
                        )
                        stage_start = stage_end
                        total += loss.detach() / 8
                        for key, value in measurements.items():
                            objectives[key] = objectives.get(key, 0.0) + value / 8
            torch.cuda.synchronize()
            local_end = time.perf_counter()
            # A single flattened reduction avoids hundreds of tiny collectives.
            gradients = torch.cat(
                [
                    (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                    for p in model.parameters()
                ]
            )
            if world > 1 and distributed:
                dist.all_reduce(gradients, op=dist.ReduceOp.SUM)
                dist.all_reduce(total, op=dist.ReduceOp.SUM)
            norm = torch.linalg.vector_norm(gradients)
            # Same global max-norm coefficient as clip_grad_norm_(..., 1.0).
            # Keep the unmodified vector for numerical comparisons; benchmark
            # clipping its copy because no optimizer consumes these gradients.
            clipped = gradients.clone().mul_(torch.clamp(1.0 / (norm + 1e-6), max=1.0))
            torch.cuda.synchronize()
            end = time.perf_counter()
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("Nonfinite gradient norm")
            gradient_sample, position = [], 0
            for parameter in model.parameters():
                length = parameter.numel()
                gradient_sample.extend(
                    gradients[position : position + min(64, length)]
                    .float()
                    .cpu()
                    .tolist()
                )
                position += length
            record = dict(
                seconds=end - start,
                forward_backward_seconds=local_end - start,
                reduction_norm_seconds=end - local_end,
                loss=float(total),
                gradient_norm=float(norm),
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                local_objectives=objectives,
                gradient_sample=gradient_sample,
                local_stage_seconds=stage_seconds,
            )
            if world > 1 and distributed:
                timing = torch.tensor(
                    [
                        record["seconds"],
                        record["forward_backward_seconds"],
                        record["peak_allocated_gib"],
                    ],
                    device=device,
                )
                dist.all_reduce(timing, op=dist.ReduceOp.MAX)
                record.update(
                    max_rank_seconds=float(timing[0]),
                    max_rank_forward_backward_seconds=float(timing[1]),
                    max_rank_peak_allocated_gib=float(timing[2]),
                )
            del clipped
            return record

        for iteration in range(args.warmup + args.repeats):
            record = update(local_micro, model)
            record.update(iteration=iteration, warmup=iteration < args.warmup)
            record["dynamo_counters"] = {
                str(group): {str(key): int(value) for key, value in values.items()}
                for group, values in torch._dynamo.utils.counters.items()
            }
            records.append(record)
            result["variants"][variant] = records
            if rank == 0:
                atomic_json(result, args.output / "timings.json")
                print(
                    json.dumps(
                        dict(
                            variant=variant,
                            **{
                                k: v
                                for k, v in record.items()
                                if k != "gradient_sample"
                            },
                        )
                    ),
                    flush=True,
                )
        if args.profile and variant == "base" and rank == 0:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                update([local_micro[0]], model, distributed=False)
            prof.export_chrome_trace(str(args.output / "trace.json"))
            (args.output / "operators.txt").write_text(
                prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=60)
            )
            events = prof.events()
            cuda_events = sum(
                event.device_type == torch.autograd.DeviceType.CUDA for event in events
            )
            atomic_json(
                dict(
                    scope="One representative sample, no distributed reduction; profiler overhead excluded from primary timings",
                    cuda_events=cuda_events,
                    cuda_kernel_profile_valid=cuda_events > 0,
                    note="CUDA kernel activity captured"
                    if cuda_events
                    else "No CUDA events; operator table is not a GPU kernel profile",
                ),
                args.output / "profile-status.json",
            )
        if world > 1:
            dist.barrier()
        del model
        gc.collect()
        torch.cuda.empty_cache()
    forecast_module.sample_joint = original_sampler
    if rank == 0:
        atomic_json(result, args.output / "COMPLETE.json")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
