# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Short real-observation training benchmarks; never write scientific checkpoints."""

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from samudra.experiments.diffusion_latent_report import load_latent_checkpoint
from samudra.experiments.diffusion_observations import forecast_observation_crps
from samudra.experiments.observation_pilot import atomic_json, digest
from samudra.experiments.observation_training import Samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=["baseline", "channels_last"])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh benchmark output directory")
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    data = Samples(args.root / "data/observations", "cuda")
    data.use_observation_normalization()
    paths = data.paths("train")
    start = time.perf_counter()
    sample = data.load(paths[0])
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start
    results = dict(
        scope="Fixed real observation sample, full two-member CRPS optimizer updates; excludes OM4 replay, validation, I/O and checkpoint saves. Benchmark weights discarded.",
        gpu=torch.cuda.get_device_name(),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(),
        producer=os.environ["SAMUDRA_CODE_COMMIT"],
        job=os.environ.get("SLURM_JOB_ID"),
        checkpoint_sha256=digest(args.checkpoint),
        sample=str(paths[0]),
        sample_sha256=digest(paths[0]),
        forecast_leads=len(sample["month_weights"]),
        load_seconds=load_seconds,
        variants={},
    )
    for variant in args.variants:
        model, signature = load_latent_checkpoint(args.checkpoint, data)
        model.train()
        results["training_signature"] = signature
        if variant in ("channels_last", "compiled_channels_last"):
            model.to(memory_format=torch.channels_last)
        elif variant not in ("baseline", "compiled_norm", "compiled_decoder"):
            raise ValueError(variant)
        if variant in ("compiled_channels_last", "compiled_decoder"):
            model.decoder.compile()
        if variant == "compiled_norm":
            for module in model.modules():
                if isinstance(module, torch.nn.GroupNorm):
                    module.compile(fullgraph=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        records = []

        def update(index, model, optimizer):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predictions, initial = model.forecast(
                    sample["surface"],
                    sample["atmosphere"],
                    sample["contexts"],
                    data.mask,
                    sample["validity"],
                    generator=torch.Generator(device="cuda").manual_seed(1729 + index),
                    members=2,
                )
                loss = forecast_observation_crps(data, predictions, sample)
            torch.cuda.synchronize()
            forward_end = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize()
            backward_end = time.perf_counter()
            if index == 0:
                gradients = {
                    name: p.grad.detach().flatten()[:64].float().cpu().numpy()
                    for name, p in model.named_parameters()
                    if p.grad is not None
                }
                np.savez_compressed(
                    args.output / f"{variant}-gradient-sample.npz", **gradients
                )
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            torch.cuda.synchronize()
            finished = time.perf_counter()
            record = dict(
                index=index,
                warmup=index < args.warmup,
                loss=float(loss),
                gradient_norm=float(norm),
                forward_seconds=forward_end - started,
                backward_seconds=backward_end - forward_end,
                optimizer_seconds=finished - backward_end,
                total_seconds=finished - started,
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            )
            del predictions, initial, loss
            return record

        for i in range(args.warmup + args.repeats):
            record = update(i, model, optimizer)
            records.append(record)
            print(json.dumps(dict(variant=variant, **record)), flush=True)
            results["variants"][variant] = records
            atomic_json(results, args.output / "timings.json")
        if args.profile and variant == "baseline":
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                update(args.warmup + args.repeats, model, optimizer)
            prof.export_chrome_trace(str(args.output / "trace.json"))
            (args.output / "operators.txt").write_text(
                prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=40)
            )
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(results, args.output / "COMPLETE.json")


if __name__ == "__main__":
    main()
