#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Prepare frozen-parent imputations and compare three observation objectives."""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ARMS = ("control", "fast", "hybrid")


def prepare(args):
    import torch

    from samudra.experiments.diffusion_direct import load_latent_checkpoint
    from samudra.experiments.diffusion_global import make_model
    from samudra.experiments.diffusion_interventions import DIRECT_PARENT_SHA
    from samudra.experiments.diffusion_observation_fast import build_imputation
    from samudra.experiments.observation_pilot import atomic_json, atomic_torch, digest
    from samudra.experiments.observation_training import Samples
    from samudra.experiments.task_schedule import sample_indices

    started = time.monotonic()
    torch.set_num_threads(1)
    torch.manual_seed(271829)
    parent = args.root / "checkpoints/parent-16256.pt"
    if digest(parent) != DIRECT_PARENT_SHA:
        raise ValueError("Changed imputation parent")
    saved = torch.load(parent, map_location="cpu", weights_only=False)
    if saved["counts"] != {"om4": 8128, "observation": 8128} or saved["step"] != 16256:
        raise ValueError("Changed imputation parent counts")
    data = Samples(
        args.root / "data/observations",
        "cuda",
        surface_fill="zero",
        global_observations=True,
    )
    data.use_observation_normalization()
    for key, path in (
        ("observation_manifest", data.root / "SHA256SUMS"),
        ("stats", data.root / "statistics.npz"),
        ("grid", data.root / "grid.npz"),
        ("reference", args.root / "checkpoints/selection-reference.json"),
    ):
        if digest(path) != saved["contract"][key]:
            raise ValueError(f"Changed imputation input: {key}")
    training = data.paths("train")
    if len(training) != 243 or any(
        not ("1993-05" <= path.stem <= "2013-07") for path in training
    ):
        raise ValueError("Changed training-only observation cohort")
    indices = sorted(
        {
            index
            for count in range(args.updates // 2)
            for index in sample_indices(len(training), 271829, count, 8)
        }
    )
    root = args.root / (
        "cache-observation-smoke" if args.smoke else "cache-observation"
    )
    root.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        version=1,
        parent_sha256=DIRECT_PARENT_SHA,
        training_contract=saved["contract"],
        updates=args.updates,
        producer=os.environ["SAMUDRA_CODE_COMMIT"],
        cache_dtype="float16",
        sampling_steps=32,
        seed_rule="55271829 + sorted training cohort index",
        input_policy="original uncorrupted training surfaces; no validation or test examples",
        source_files={
            training[index].name: digest(training[index]) for index in indices
        },
    )
    if (root / "protocol.json").exists() and json.loads(
        (root / "protocol.json").read_text()
    ) != protocol:
        raise ValueError("Changed cache protocol")
    atomic_json(protocol, root / "protocol.json")
    model = (
        make_model(data.grid["names"].tolist(), "cuda", steps=32)
        .eval()
        .requires_grad_(False)
    )
    load_latent_checkpoint(model, saved)
    del saved
    model.decoder.compile()
    files = {}
    seconds_generated = 0.0
    for index in indices:
        path = training[index]
        cache = root / f"{path.stem}.pt"
        stamp = root / f"{path.stem}.input.json"
        identity = dict(
            parent_sha256=DIRECT_PARENT_SHA,
            protocol_sha256=digest(root / "protocol.json"),
            source_sha256=protocol["source_files"][path.name],
            seed=55271829 + index,
        )
        if cache.exists() and stamp.exists():
            previous = json.loads(stamp.read_text())
            if previous["inputs"] != identity or previous["sha256"] != digest(cache):
                raise ValueError(f"Changed existing imputation: {path.name}")
            files[cache.name] = previous["sha256"]
            continue
        before = time.monotonic()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            entry = build_imputation(
                model, data, data.load(path), seed=identity["seed"]
            )
        entry = {
            key: value.to(dtype=torch.float16)
            if torch.is_tensor(value) and value.is_floating_point()
            else value
            for key, value in entry.items()
        }
        if not all(
            bool(torch.isfinite(value).all())
            for value in entry.values()
            if torch.is_tensor(value)
        ):
            raise FloatingPointError("Imputation storage conversion became nonfinite")
        atomic_torch(entry, cache)
        elapsed = time.monotonic() - before
        seconds_generated += elapsed
        files[cache.name] = digest(cache)
        atomic_json(
            dict(
                inputs=identity,
                sha256=files[cache.name],
                seconds=elapsed,
                bytes=cache.stat().st_size,
            ),
            stamp,
        )
        print(
            json.dumps(
                dict(
                    event="imputation",
                    name=path.stem,
                    seconds=elapsed,
                    complete=len(files),
                    total=len(indices),
                )
            ),
            flush=True,
        )
    torch.cuda.synchronize()
    if (root / "COMPLETE.json").exists():
        previous = json.loads((root / "COMPLETE.json").read_text())
        if previous["files"] != files or any(
            previous[key] != value for key, value in protocol.items()
        ):
            raise ValueError("Completed cache changed")
        return  # Keep the exact manifest hash used by existing training protocols.
    atomic_json(
        dict(
            **protocol,
            files=files,
            seconds=time.monotonic() - started,
            seconds_generated=seconds_generated,
            total_bytes=sum((root / name).stat().st_size for name in files),
            complete=True,
        ),
        root / "COMPLETE.json",
    )


def smoke_readout(args):
    import math

    import torch

    from samudra.experiments.diffusion_direct import load_latent_checkpoint
    from samudra.experiments.diffusion_global import make_model, rng
    from samudra.experiments.diffusion_spatial import forecast_crps
    from samudra.experiments.observation_pilot import atomic_json, digest
    from samudra.experiments.observation_training import Samples

    started = time.monotonic()
    torch.set_num_threads(1)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    expected = 8128 if args.checkpoint.name == "parent-16256.pt" else 8132
    if saved["counts"] != {"om4": expected, "observation": expected}:
        raise ValueError("Unexpected qualification checkpoint counts")
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
            raise ValueError("Qualification data differs")
    model = make_model(data.grid["names"].tolist(), "cuda", steps=32).eval()
    load_latent_checkpoint(model, saved)
    path = data.paths("validation")[0]
    sample = data.load(path)
    norms = []
    handle = model.processor.register_forward_hook(
        lambda module, inputs, output: norms.append(
            float(output.float().square().mean().sqrt())
        )
    )
    seed = 4041729
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictions, initial = model.forecast(
                sample["surface"],
                sample["atmosphere"],
                sample["contexts"],
                data.mask,
                sample["validity"],
                generator=rng(data.device, seed),
                members=2,
            )
            score = float(forecast_crps(data, predictions, sample))
        if (
            not bool(torch.isfinite(predictions).all())
            or not bool(torch.isfinite(initial).all())
            or not math.isfinite(score)
            or not all(math.isfinite(value) for value in norms)
        ):
            raise FloatingPointError("Nonfinite qualification sampler output")
    finally:
        handle.remove()
    atomic_json(
        dict(
            complete=True,
            checkpoint_sha256=digest(args.checkpoint),
            counts=saved["counts"],
            input=str(path),
            input_sha256=digest(path),
            seed=seed,
            members=2,
            sampling_steps=32,
            forecast_crps=score,
            latent_rms=norms,
            seconds=time.monotonic() - started,
        ),
        args.readout_output,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=128)
    parser.add_argument("--hours", type=float, default=7)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--smoke-readout", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--readout-output", type=Path)
    args = parser.parse_args()
    if args.smoke:
        args.updates = 8
    elif args.updates != 128:
        raise ValueError("Production requires exactly 128 mixed updates")
    if args.smoke_readout:
        if args.checkpoint is None or args.readout_output is None:
            raise ValueError("Qualification requires checkpoint and output")
        smoke_readout(args)
        return
    if args.prepare:
        prepare(args)
        return
    started = time.monotonic()
    logroot = args.root / "logs" / os.environ.get("SLURM_JOB_ID", "local")
    logroot.mkdir(parents=True, exist_ok=True)

    def environment(gpu):
        return dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=str(gpu),
            LOCAL_RANK="0",
            WORLD_SIZE="1",
            RANK="0",
            TORCHINDUCTOR_CACHE_DIR=str(args.root / "cache" / f"gpu-{gpu}"),
            TRITON_CACHE_DIR=str(args.root / "cache" / f"triton-{gpu}"),
        )

    def prep_run():
        before = time.monotonic()
        command = [
            sys.executable,
            __file__,
            "--root",
            str(args.root),
            "--updates",
            str(args.updates),
            "--prepare",
        ]
        if args.smoke:
            command.append("--smoke")
        with (logroot / "cache-preparation.log").open("a") as log:
            subprocess.run(
                command,
                env=environment(3),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        if args.smoke:
            with (logroot / "parent-readout.log").open("a") as log:
                subprocess.run(
                    [
                        sys.executable,
                        __file__,
                        "--root",
                        str(args.root),
                        "--smoke-readout",
                        "--checkpoint",
                        str(args.root / "checkpoints/parent-16256.pt"),
                        "--readout-output",
                        str(logroot / "parent-readout.json"),
                    ],
                    env=environment(3),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        return dict(seconds=time.monotonic() - before, gpu=3)

    def arm_run(gpu, arm, cache_future):
        before = time.monotonic()
        durations = {}
        try:
            if arm != "control":
                cache_future.result()
            durations["wait_for_cache"] = time.monotonic() - before
            checkpoint = (
                args.root
                / ("smoke" if args.smoke else "runs")
                / arm
                / f"step-{args.updates:04d}.pt"
            )
            with (logroot / f"{arm}.log").open("a") as log:

                def run(label, command):
                    task_started = time.monotonic()
                    subprocess.run(
                        [sys.executable, *command],
                        env=environment(gpu),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                    durations[label] = time.monotonic() - task_started

                command = [
                    "-m",
                    "samudra.experiments.diffusion_interventions",
                    "--campaign",
                    "obs-fast",
                    "--root",
                    str(args.root),
                    "--arm",
                    arm,
                    "--updates",
                    str(args.updates),
                    "--hours",
                    str(0.8 if args.smoke else args.hours),
                ]
                if args.smoke:
                    command += ["--smoke"]
                run("training", command)
                if args.smoke:
                    run(
                        "strict_reload_and_readout",
                        [
                            __file__,
                            "--root",
                            str(args.root),
                            "--smoke-readout",
                            "--checkpoint",
                            str(checkpoint),
                            "--readout-output",
                            str(logroot / f"{arm}-readout.json"),
                        ],
                    )
                else:
                    common = [
                        "--data",
                        str(args.root / "data/observations"),
                        "--checkpoint",
                        str(checkpoint),
                        "--reference",
                        str(args.root / "checkpoints/selection-reference.json"),
                        "--steps",
                        "32",
                    ]
                    output = args.root / "evaluation" / arm
                    run(
                        "validation",
                        [
                            "-m",
                            "samudra.experiments.diffusion_global_evaluate",
                            *common,
                            "--output",
                            str(output / "validation"),
                        ],
                    )
                    run(
                        "native",
                        [
                            "-m",
                            "samudra.experiments.diffusion_intervention_native",
                            "--root",
                            str(args.root),
                            "--checkpoint",
                            str(checkpoint),
                            "--output",
                            str(output / "native"),
                        ],
                    )
                    run(
                        "annual",
                        [
                            "-m",
                            "samudra.experiments.diffusion_global_annual",
                            *common,
                            "--annual-data",
                            str(args.root / "data/annual_observations"),
                            "--expected-observation-updates",
                            "8192",
                            "--expected-om4-updates",
                            "8192",
                            "--output",
                            str(output / "annual"),
                        ],
                    )
            return dict(
                arm=arm,
                success=True,
                gpu=gpu,
                timings=durations,
                total_seconds=time.monotonic() - before,
            )
        except subprocess.CalledProcessError as error:
            return dict(
                arm=arm,
                success=False,
                exit_code=error.returncode,
                gpu=gpu,
                timings=durations,
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        cache = pool.submit(prep_run)
        futures = [
            pool.submit(arm_run, gpu, arm, cache) for gpu, arm in enumerate(ARMS)
        ]
        results = [future.result() for future in futures]
        preparation = cache.result()
    total_seconds = time.monotonic() - started
    report = dict(
        arms=results,
        preparation=preparation,
        wall_seconds=total_seconds,
        allocated_gpu_hours_excluding_job_startup=4 * total_seconds / 3600,
        accounting="All four GPUs count for the entire allocation, including cache wait and idle slots; final Slurm accounting remains authoritative",
    )
    (logroot / "RESULTS.json").write_text(json.dumps(report, indent=2) + "\n")
    if not all(result["success"] for result in results):
        raise RuntimeError(f"Some arms failed; inspect {logroot}/RESULTS.json")


if __name__ == "__main__":
    main()
