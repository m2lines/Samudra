# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Exact 8k/8k mixed exposure with synchronous gradient accumulation and resume."""

import argparse
import datetime
import gc
import json
import os
import signal
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from samudra.experiments.diffusion_correlated_pilot import build_wave
from samudra.experiments.diffusion_evaluation import evaluate_point_metrics
from samudra.experiments.diffusion_global import (
    make_model,
    native_objective,
    observation_parts,
)
from samudra.experiments.diffusion_latent_train import gradient_norm
from samudra.experiments.observation_pilot import atomic_json, atomic_torch, digest
from samudra.experiments.observation_training import Samples
from samudra.experiments.task_schedule import TaskSchedule, sample_indices

MILESTONES = (10, 25, 50, 100, 250, 500, 1000, 2000, 4000, 6000, 8000)


def emit(event):
    print(json.dumps(event), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--qualify", action="store_true")
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--compile-decoder", action="store_true")
    parser.add_argument("--invocation-hours", type=float, default=11.5)
    parser.add_argument("--max-new-updates", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--evaluate", type=Path)
    parser.add_argument("--eval-steps", type=int, default=32)
    args = parser.parse_args()
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if 8 % world:
        raise ValueError("World size must divide effective batch eight")
    device = torch.device("cuda", local)
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    if world > 1:
        dist.init_process_group("nccl", timeout=datetime.timedelta(hours=1))
    args.output.mkdir(parents=True, exist_ok=True)
    data = Samples(
        args.root / "data/observations",
        device,
        surface_fill="zero",
        global_observations=True,
    )
    data.use_observation_normalization()
    training, validation = data.paths("train"), data.paths("validation")
    if (len(training), len(validation), len(data.paths("test"))) != (243, 9, 96):
        raise ValueError("Observation cohorts changed")
    contract = dict(
        version=1,
        seed=1729,
        schedule="quadratic mixed 8000/8000",
        effective_batch=8,
        decoder_width=192,
        latent_width=128,
        processor_depth=4,
        sampling_steps=32,
        noise_correlation=0.5,
        spatial_weight=0.5,
        reconstruction_weight=0.1,
        completion_weight=0.1,
        global_observations=True,
        surface_fill="zero",
        task_conditioning="identity input adapters for encoder and processor",
        learning_rate=1e-4,
        weight_decay=0.01,
        clip=1.0,
        training_members=2,
        native_loss="channel-balanced clean-field denoising plus spatial differences",
        observation_loss="group-balanced fair CRPS plus spatial-increment fair CRPS",
        observation_manifest=digest(data.root / "SHA256SUMS"),
        grid=digest(data.root / "grid.npz"),
        stats=digest(data.root / "statistics.npz"),
        reference=digest(args.reference),
        compile_decoder=args.compile_decoder,
        producer=os.environ["SAMUDRA_CODE_COMMIT"],
    )
    if not args.qualify and not args.evaluate:
        if args.qualification is None:
            raise ValueError("Qualification required")
        q = json.loads(args.qualification.read_text())
        if q["contract"] != contract or not q["qualified"]:
            raise ValueError("Qualification contract differs")
    protocol = args.output / "protocol.json"
    if protocol.exists() and json.loads(protocol.read_text()) != contract:
        raise ValueError("Output protocol differs")
    if rank == 0:
        atomic_json(contract, protocol)
    # Existing Rust loader retains exactly the comparator's native windows.
    wave = None
    if not args.evaluate:
        loader = SimpleNamespace(
            root=args.root, output=args.output / f"rank-{rank}", export_only=True
        )
        wave = build_wave(loader, data)
        del wave.model
        gc.collect()
        wave.prepare(wave.trainset)
        if list(wave.names) != list(data.grid["names"]):
            raise ValueError("Native channel order differs")
    torch.manual_seed(1729)
    model = make_model(data.grid["names"].tolist(), device)
    if args.compile_decoder:
        model.decoder.compile()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    schedule = TaskSchedule(8000, 8000, "mixed")
    completed = 0
    last = args.output / "last.pt"
    if last.exists() and not args.qualify and not args.evaluate:
        saved = torch.load(last, map_location="cpu", weights_only=False)
        if saved["contract"] != contract:
            raise ValueError("Checkpoint contract differs")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        completed = saved["step"]
        if saved["counts"] != schedule.counts(completed):
            raise ValueError("Saved exposure differs")
        # Training draws are task/microbatch keyed; all ranks restore identical
        # global RNG as well. The model has no dropout or running batch statistics.
        torch.set_rng_state(saved["cpu_rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"], device)
        del saved
    reference = json.loads(args.reference.read_text())
    if args.evaluate:
        saved = torch.load(args.evaluate, map_location="cpu", weights_only=False)
        if saved["contract"] != contract:
            raise ValueError("Evaluation checkpoint contract differs")
        model.load_state_dict(saved["model"], strict=True)
        model.sampling_steps = args.eval_steps
        model.eval()
        with torch.no_grad():
            metrics, score = evaluate_point_metrics(
                model, data, validation, reference, members=8
            )
        atomic_json(
            dict(
                step=saved["step"],
                counts=saved["counts"],
                checkpoint_sha256=digest(args.evaluate),
                sampling_steps=args.eval_steps,
                score=score,
                metrics=metrics,
            ),
            args.output / f"validation-{saved['step']}-{args.eval_steps}.json",
        )
        emit(dict(event="validation", step=saved["step"], score=score))
        return

    stop = False

    def request_stop(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    def save(snapshot=False):
        if world > 1:
            dist.barrier()
        if rank == 0:
            payload = dict(
                contract=contract,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                step=completed,
                counts=schedule.counts(completed),
                cpu_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(device),
            )
            atomic_torch(payload, last)
            if snapshot:
                atomic_torch(
                    dict(
                        contract=contract,
                        model=model.state_dict(),
                        step=completed,
                        counts=schedule.counts(completed),
                    ),
                    args.output / f"step-{completed:05d}.pt",
                )
            atomic_json(
                dict(
                    step=completed,
                    counts=schedule.counts(completed),
                    complete=completed == 16000,
                ),
                args.output / "state.json",
            )
        if world > 1:
            dist.barrier()

    def update(task, count):
        assert wave is not None
        started = time.monotonic()
        seed = 1729 + (0 if task == "observation" else 100000)
        size = len(training) if task == "observation" else len(wave.trainset)
        ids = sample_indices(size, seed, count, 8)
        optimizer.zero_grad(set_to_none=True)
        measurements: dict[str, float] = {}
        total = 0.0
        micro_seconds = []
        model.train()
        for micro in range(rank, 8, world):
            micro_started = time.monotonic()
            index = ids[micro]
            noise_seed = seed + 1000000 + count * 8 + micro
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if task == "observation":
                    sample = data.load(training[index])
                    for label, loss, parts in observation_parts(
                        model, data, sample, noise_seed
                    ):
                        if not bool(torch.isfinite(loss)):
                            raise FloatingPointError(label)
                        (loss / 8).backward()
                        total += float(loss.detach()) / 8
                        for key, value in parts.items():
                            measurements[key] = measurements.get(key, 0.0) + value / 8
                else:
                    coverage = data.load(training[(count * 8 + micro) % len(training)])[
                        "validity"
                    ][:, :19]
                    loss = native_objective(model, wave, index, coverage, noise_seed)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("native")
                    (loss / 8).backward()
                    total += float(loss.detach()) / 8
            micro_seconds.append(time.monotonic() - micro_started)
        # Sum each rank's 1/8-scaled microbatch gradients exactly once. Unused
        # ERA5/source adapters keep grad=None, so AdamW does not decay them.
        if world > 1:
            for parameter in model.parameters():
                if parameter.grad is not None:
                    dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        norms = {
            name: gradient_norm(module)
            for name, module in (
                ("encoder", model.encoder),
                ("processor", model.processor),
                ("decoder", model.decoder),
            )
        }
        if not all(v > 0 for v in norms.values()):
            raise ValueError(f"Missing gradients: {norms}")
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        torch.cuda.synchronize()
        event = dict(
            event="update",
            task=task,
            task_update=count + 1,
            step=completed + 1,
            seconds=time.monotonic() - started,
            loss_rank_contribution=total,
            parts_rank_contribution=measurements,
            gradient_norm=float(norm),
            gradient_norms=norms,
            peak_gib=torch.cuda.max_memory_allocated() / 2**30,
            world_size=world,
            micro_seconds=micro_seconds,
        )
        if rank == 0:
            emit(event)
            with (args.output / "events.jsonl").open("a") as stream:
                stream.write(json.dumps(event) + "\n")
        return event

    if args.qualify:
        if world != 1:
            raise ValueError("Initial qualification runs on one GPU")
        sample = data.load(training[0])
        if bool(sample["surface"][~sample["validity"].bool()].count_nonzero()):
            raise ValueError("Missing input not zero normalized")
        if not bool((data.area > 0).all()):
            raise ValueError("Latitude loss cutoff remains")
        measurements = [update("om4", 0), update("observation", 0)]
        completed = 1
        save()
        saved = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value.cpu(), saved["model"][key], atol=0, rtol=0)
        # The checkpoint roundtrip is qualified; actual production restart uses
        # the same loader above. Sample/noise streams depend only on task counts.
        atomic_json(
            dict(
                qualified=True,
                contract=contract,
                measurements=measurements,
                resume="strict model/optimizer reload; task-keyed inputs and noise",
            ),
            args.output / "QUALIFIED.json",
        )
        return

    started, first = time.monotonic(), completed
    if not last.exists():
        save(snapshot=True)
    while completed < schedule.total:
        task = schedule.task(completed)
        count = schedule.counts(completed)[task]
        update(task, count)
        completed += 1
        milestone = task == "observation" and count + 1 in MILESTONES
        due = stop or time.monotonic() - started >= args.invocation_hours * 3600
        due |= (
            args.max_new_updates is not None
            and completed - first >= args.max_new_updates
        )
        flag = torch.tensor(int(due), device=device)
        if world > 1:
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if (
            completed % args.checkpoint_every == 0
            or milestone
            or bool(flag)
            or completed == 16000
        ):
            save(snapshot=milestone or completed == 16000)
        if bool(flag):
            break
    if rank == 0:
        emit(
            dict(
                event="invocation_finished",
                step=completed,
                counts=schedule.counts(completed),
                complete=completed == 16000,
            )
        )
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
