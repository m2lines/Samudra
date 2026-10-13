#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Qualified common forks and isolated Stage-1 stability diagnostics.

No diagnostic weights are promoted into the Stage-2 treatment forks.
"""

import argparse
import contextlib
import importlib
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from samudra.experiments.early_fine_training import EarlyPilot, deterministic_replay
from samudra.experiments.observation_joint import (
    assert_exact_state,
    om4_objective,
    qualification_contract,
)
from samudra.experiments.observation_metrics import selection_score
from samudra.experiments.observation_model import ObservationTransfer
from samudra.experiments.observation_pilot import Pilot, atomic_json, digest
from samudra.experiments.observation_training import Samples
from samudra.experiments.task_schedule import TaskSchedule, sample_indices
from scripts.prepare_early_fine_cooldown import audit_arguments
from scripts.run_early_fine_wave import command

ARMS = ("U-global", "U-multitask-early")


@contextlib.contextmanager
def preserve_execution(model):
    """Diagnostics must not consume training RNG or leave a different mode."""
    cpu, cuda, numpy = (
        torch.get_rng_state(),
        torch.cuda.get_rng_state(),
        np.random.get_state(),
    )
    modes = [(m, m.training) for m in model.modules()]
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        torch.cuda.set_rng_state(cuda)
        np.random.set_state(numpy)
        for module, mode in modes:
            module.training = mode


def check_migration(old, new):
    """Only reporting, paths and the explicitly declared control routing change."""
    old = dict(old)
    new = dict(new)
    for key in (
        "selection_metric",
        "wandb_mode",
        "model_only_milestones",
        "milestone_steps",
        "validate_every",
        "name",
        "joint_probe",
    ):
        old.pop(key, None)
        new.pop(key, None)
    # New evaluator option has no effect on the legacy, single-grid task.
    for row in (old, new):
        if row.get("iap_grid") is None:
            row.pop("iap_grid", None)
    audit_arguments(old, new)


def check_runtime(root, config):
    if config["producer"] != os.environ["SAMUDRA_CODE_COMMIT"]:
        raise ValueError("Wrong pinned producer")
    if digest(root / "source.tar") != config["archive_sha256"]:
        raise ValueError("Source archive changed")
    runtime = json.loads((root / "runtime-contract.json").read_text())
    for name, expected in runtime["modules"].items():
        module = importlib.import_module(name)
        if getattr(module, "__version__", None) != expected["version"]:
            raise ValueError("Runtime version drift: " + name)
        for path, sha in expected["binaries"].items():
            if digest(path) != sha:
                raise ValueError("Runtime binary drift: " + path)


def close(pilot):
    pilot.early.prefetch.close()
    pilot.early.pool.shutdown(wait=True, cancel_futures=True)
    pilot.observation_warm_pool.shutdown(wait=True, cancel_futures=True)
    if pilot.wandb:
        pilot.wandb.finish()


def make_pilot(root, arm, out, source, checksum):
    probe = root / ("probe-" + arm)
    q = json.loads((probe / "JOINT_QUALIFIED.json").read_text())
    io = json.loads((probe / "IO_EQUIVALENT.json").read_text())
    if not (
        q["resume_verified"]
        and q["resume_evidence"]["deterministic_reference_bitwise_exact"]
    ):
        raise ValueError("Missing exact replay qualification")
    if not (io["cpu_arrays_exact"] and io["observation_prepared_and_coverage_exact"]):
        raise ValueError("Missing exact I/O qualification")
    options = json.loads((probe / "manifest.json").read_text())["arguments"]
    options.update(
        output=str(out),
        name="stability-" + arm + "-" + out.name,
        joint_probe=False,
        om4_updates=2000,
        joint_steps=2000,
        joint_qualification=str(probe / "JOINT_QUALIFIED.json"),
        validate_every=100,
        joint_hours=24,
        milestone_steps=[],
        model_only_milestones=False,
        wandb_mode="online",
    )
    if options["recent_only"] != (arm == "U-global"):
        raise ValueError("Incorrect early/recent routing")
    args = SimpleNamespace(**options)
    if q["contract"] != qualification_contract(args):
        raise ValueError("Qualification contract differs")
    pilot = EarlyPilot(args)
    if not pilot.wandb or not pilot.wandb.url:
        raise RuntimeError("Online W&B run was not established")
    atomic_json({"url": pilot.wandb.url}, out / "WANDB.json")
    if digest(source) != checksum:
        raise ValueError("Source checkpoint changed")
    saved = torch.load(source, map_location="cpu", weights_only=False)
    if not pilot.resume.exists():
        check_migration(saved["manifest"]["arguments"], options)
        for key in (
            "normalization_mode",
            "effective_mean",
            "effective_std",
            "data_manifest_sha256",
            "grid_sha256",
            "statistics_sha256",
            "training_months",
            "validation_months",
        ):
            assert_exact_state(saved["manifest"][key], pilot.manifest[key])
        if saved["optimizer"] is None or saved["task_counts"] != pilot.schedule.counts(
            saved["global_step"]
        ):
            raise ValueError("Missing optimizer or incorrect source task counts")
        pilot.model.load_state_dict(saved["model"], strict=True)
        pilot.optimizer.load_state_dict(saved["optimizer"])
        pilot.completed = saved["global_step"]
        torch.set_rng_state(saved["rng_cpu"])
        torch.cuda.set_rng_state(saved["rng_cuda"])
        np.random.set_state(saved["rng_numpy"])
        pilot.checkpoint(pilot.resume)
        restored = torch.load(pilot.resume, map_location="cpu", weights_only=False)
        for key in (
            "model",
            "optimizer",
            "global_step",
            "step",
            "phase",
            "task_counts",
            "rng_cpu",
            "rng_cuda",
            "rng_numpy",
        ):
            assert_exact_state(restored[key], saved[key])
        atomic_json(
            dict(
                source=str(source),
                source_sha256=checksum,
                source_producer=saved["manifest"]["code_commit"],
                producer=pilot.manifest["code_commit"],
                exact_state_keys=[
                    "model",
                    "optimizer",
                    "global_step",
                    "step",
                    "phase",
                    "task_counts",
                    "rng_cpu",
                    "rng_cuda",
                    "rng_numpy",
                ],
                changes="Reporting/provenance paths and RMSE-only selection; wall-time accounting starts anew. Objective, task/sample sequence and optimizer unchanged.",
                old_arguments=saved["manifest"]["arguments"],
                new_arguments=options,
            ),
            out / "MIGRATION.json",
        )
    else:
        receipt = json.loads((out / "MIGRATION.json").read_text())
        if receipt["source_sha256"] != checksum:
            raise ValueError("Resume source changed")
    if digest(source) != checksum:
        raise ValueError("Source mutated during load")
    del saved
    return pilot


def validation(pilot, label):
    with preserve_execution(pilot.model):
        metrics = pilot.evaluate(pilot.validation)
    reference = json.loads(Path(pilot.args.selection_reference).read_text())
    if reference["control"]["origins"] != [p.stem for p in pilot.validation]:
        raise ValueError("Validation origins differ from frozen reference")
    value = selection_score(metrics, reference["control"], [], metric="rmse-only")
    record = dict(
        global_step=pilot.completed,
        metrics=metrics,
        score=value,
        task_counts=pilot.schedule.counts(pilot.completed),
    )
    atomic_json(record, pilot.out / (label + ".json"))
    pilot.emit(
        dict(
            event="stability_validation",
            global_step=pilot.completed,
            **{"validation/rmse_blend": value},
        )
    )
    return record


def frozen_loss(pilot, task, index=0):
    seed = pilot.args.seed + 1000000 + index
    if task == "observation":
        sample = pilot.data.load(pilot.training[index])
        sample["mask_seed"] = seed
        return pilot.objective(sample, "joint")
    coverage = pilot.observation_coverage(pilot.training[index])
    if task == "recent":
        return om4_objective(
            pilot.model,
            pilot.om4,
            [index],
            pilot.args.reconstruction_weight,
            seed,
            coverage,
            pilot.args.completion_weight,
        )
    # Isolate the early branch without altering the real sampler/counters.
    completed, recent_only = pilot.completed, pilot.args.recent_only
    try:
        pilot.completed = 1  # the 4k schedule has consumed one OM4 update here
        pilot.args.recent_only = False
        return pilot.om4_objective(
            pilot.model,
            pilot.om4,
            [index],
            pilot.args.reconstruction_weight,
            seed,
            coverage,
            pilot.args.completion_weight,
        )
    finally:
        pilot.completed, pilot.args.recent_only = completed, recent_only


def probes(pilot):
    with (
        preserve_execution(pilot.model),
        torch.no_grad(),
        torch.autocast("cuda", dtype=torch.bfloat16),
    ):
        pilot.model.eval()
        return {
            task: float(frozen_loss(pilot, task))
            for task in ("observation", "recent", "early")
        }


def gradients(pilot, label):
    parameters = {
        component: [
            p
            for name, p in getattr(pilot.model, component).named_parameters()
            if "input_adapters" not in name
        ]
        for component in ("initializer", "evolution")
    }
    vectors, tasks = {}, {}
    with preserve_execution(pilot.model):
        pilot.model.train()
        for task in ("observation", "recent", "early"):
            pilot.model.zero_grad(set_to_none=True)
            losses = []
            for index in (0, 33, 101, 211):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = frozen_loss(pilot, task, index)
                (loss / 4).backward()
                losses.append(float(loss.detach()))
            vectors[task] = {
                component: torch.cat(
                    [
                        (
                            p.grad.detach().cpu()
                            if p.grad is not None
                            else torch.zeros_like(p, device="cpu")
                        ).flatten()
                        for p in params
                    ]
                )
                for component, params in parameters.items()
            }
            tasks[task] = dict(
                losses=losses,
                norms={k: float(v.norm()) for k, v in vectors[task].items()},
            )
        cosines = {}
        for left, right in (
            ("observation", "recent"),
            ("observation", "early"),
            ("recent", "early"),
        ):
            cosines[left + "/" + right] = {
                name: float(
                    torch.dot(vectors[left][name], vectors[right][name])
                    / (
                        vectors[left][name].norm() * vectors[right][name].norm()
                    ).clamp_min(1e-30)
                )
                for name in parameters
            }
        pilot.model.zero_grad(set_to_none=True)
    atomic_json(
        dict(
            global_step=pilot.completed,
            tasks=tasks,
            cosines=cosines,
            indices=[0, 33, 101, 211],
            shared_parameters_only=True,
        ),
        pilot.out / (label + ".json"),
    )


def fork(root, arm, config):
    source = config["sources"][arm]
    out = root / arm / "fork"
    if (out / "FORK_COMPLETE.json").exists():
        receipt = json.loads((out / "FORK_COMPLETE.json").read_text())
        if digest(out / "joint-last.pt") != receipt["sha256"]:
            raise ValueError("Completed common fork changed")
        return
    pilot = make_pilot(root, arm, out, Path(source["checkpoint"]), source["sha256"])
    try:
        if not 2667 <= pilot.completed <= 3000:
            raise ValueError("Wrong common fork starting step")
        while pilot.completed < 3000:
            pilot.train_update()
        pilot.checkpoint(pilot.resume)
        validation(pilot, "fork-validation")
        atomic_json(
            dict(
                sha256=digest(pilot.resume),
                global_step=3000,
                task_counts=pilot.schedule.counts(3000),
            ),
            out / "FORK_COMPLETE.json",
        )
    finally:
        close(pilot)


def trace(root, arm, trial):
    common = root / arm / "fork"
    receipt = json.loads((common / "FORK_COMPLETE.json").read_text())
    out = root / arm / f"trace-{trial}"
    if (out / "COMPLETE.json").exists():
        return
    pilot = make_pilot(root, arm, out, common / "joint-last.pt", receipt["sha256"])
    try:
        if not (out / "gradients-start.json").exists():
            gradients(pilot, "gradients-start")
        if pilot.completed == 3000:
            validation(pilot, "validation-03000")
        while pilot.completed < 3100:
            before = {n: p.detach().clone() for n, p in pilot.model.named_parameters()}
            task = pilot.schedule.task(pilot.completed)
            count = pilot.schedule.counts(pilot.completed)[task]
            ids = sample_indices(
                len(pilot.training)
                if task == "observation"
                else len(pilot.om4.trainset),
                pilot.args.seed + (0 if task == "observation" else 100000),
                count,
                pilot.args.accumulate,
            )
            pre = probes(pilot)
            pilot.loss_components = []
            pilot.loss_observer = pilot.loss_components.append
            try:
                pilot.train_update()
            finally:
                pilot.loss_observer = None
            update_norm = math.sqrt(
                sum(
                    float((p.detach() - before[n]).square().sum())
                    for n, p in pilot.model.named_parameters()
                )
            )
            del before
            post = probes(pilot)
            components = (
                {
                    key: float(
                        torch.stack([row[key] for row in pilot.loss_components]).mean()
                    )
                    for key in pilot.loss_components[0]
                }
                if pilot.loss_components
                else {}
            )
            mask_seed_base = (
                pilot.args.seed
                + (1000000 if task == "observation" else 1100000)
                + count * pilot.args.accumulate
            )
            masks = list(range(mask_seed_base, mask_seed_base + pilot.args.accumulate))
            early = task == "om4" and count % 2 and not pilot.args.recent_only
            origins = (
                [pilot.training[i].stem for i in ids]
                if task == "observation"
                else [
                    int(
                        pilot.early.origins[
                            int(
                                np.random.default_rng(seed).integers(
                                    len(pilot.early.origins)
                                )
                            )
                        ]
                    )
                    for seed in masks
                ]
                if early
                else ids
            )
            pilot.emit(
                dict(
                    event="stability_update",
                    global_step=pilot.completed,
                    task=task,
                    sample_indices=ids,
                    sampled_origins=origins,
                    mask_seeds=masks,
                    source="early" if early else task,
                    observation_loss_components=components,
                    task_count_before=count,
                    parameter_update_norm=update_norm,
                    fixed_training_losses_before=pre,
                    fixed_training_losses_after=post,
                )
            )
            if pilot.completed % 10 == 0:
                validation(pilot, f"validation-{pilot.completed:05d}")
                pilot.checkpoint(pilot.resume)
        gradients(pilot, "gradients-end")
        pilot.checkpoint(pilot.resume)
        atomic_json(
            dict(
                global_step=pilot.completed,
                trial=trial,
                source_sha256=receipt["sha256"],
                diagnostic_weights_only=True,
                sha256=digest(pilot.resume),
            ),
            out / "COMPLETE.json",
        )
    finally:
        close(pilot)


def repeat(root, arm, group):
    common = root / arm / "fork"
    receipt = json.loads((common / "FORK_COMPLETE.json").read_text())
    checkpoint = common / "joint-last.pt"
    if digest(checkpoint) != receipt["sha256"]:
        raise ValueError("Common fork changed")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    options = saved["manifest"]["arguments"]
    pilot = Pilot.__new__(
        Pilot
    )  # evaluation only, same construction as observation_evaluate
    pilot.args = SimpleNamespace(**options)
    pilot.data = Samples(
        options["data"], "cuda", options["surface_fill"], global_observations=True
    )
    pilot.data.use_observation_normalization()
    pilot.data.cache_prepared = True
    pilot.model = ObservationTransfer.from_arguments(
        pilot.data.grid["names"].tolist(), options
    ).cuda()
    pilot.model.load_state_dict(saved["model"], strict=True)
    reference = json.loads(Path(options["selection_reference"]).read_text())
    paths = pilot.data.paths("validation")
    if len(paths) != 9 or [p.stem for p in paths] != reference["control"]["origins"]:
        raise ValueError("Incorrect validation cohort")
    out = root / arm / "repeatability"
    out.mkdir(parents=True, exist_ok=True)
    trials = (0, 1, 2) if group == 0 else (group + 2,)
    for trial in trials:
        marker = out / f"trial-{trial}.json"
        if marker.exists():
            continue
        deterministic = group in (3, 4)
        with deterministic_replay() if deterministic else contextlib.nullcontext():
            metrics = pilot.evaluate(paths, export=out / f"trial-{trial}.npz")
        assert_exact_state(pilot.model.state_dict(), saved["model"])
        value = selection_score(metrics, reference["control"], [], metric="rmse-only")
        atomic_json(
            dict(
                trial=trial,
                process_group=group,
                deterministic=deterministic,
                metrics=metrics,
                score=value,
                source_sha256=receipt["sha256"],
                unchanged_model_state=True,
            ),
            marker,
        )
    if digest(checkpoint) != receipt["sha256"]:
        raise ValueError("Read-only diagnostic mutated source")


def audit(root, config):
    runtime = json.loads((root / "runtime-contract.json").read_text())
    if digest(config["image"]) != runtime["image_sha256"]:
        raise ValueError("Container image drift")
    evidence = {}
    for arm, source in config["sources"].items():
        if digest(source["checkpoint"]) != source["sha256"]:
            raise ValueError("Source checkpoint drift: " + arm)
        saved = torch.load(source["checkpoint"], map_location="cpu", weights_only=False)
        if saved["global_step"] != 2667 or saved["optimizer"] is None:
            raise ValueError("Source is not a restartable 2667 checkpoint")
        assert_exact_state(
            saved["task_counts"], TaskSchedule(2000, 2000, "mixed").counts(2667)
        )
        for key, name in (
            ("data_manifest_sha256", "SHA256SUMS"),
            ("grid_sha256", "grid.npz"),
            ("statistics_sha256", "statistics.npz"),
        ):
            if digest(Path(config["observations"]) / name) != saved["manifest"][key]:
                raise ValueError("Observation lineage drift: " + name)
        evidence[arm] = dict(
            source=source,
            source_producer=saved["manifest"]["code_commit"],
            global_step=2667,
        )
        del saved
    reference = json.loads((root / "selection-reference.json").read_text())
    if reference["data_manifest_sha256"] != digest(
        Path(config["observations"]) / "SHA256SUMS"
    ):
        raise ValueError("RMSE reference data mismatch")
    if len(reference["control"]["origins"]) != 9:
        raise ValueError("Wrong validation reference cohort")
    atomic_json(
        dict(
            config_sha256=digest(root / "paths.json"),
            runtime_sha256=digest(root / "runtime-contract.json"),
            sources=evidence,
            image_sha256=runtime["image_sha256"],
        ),
        root / "SOURCE_AUDITED.json",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument(
        "--stage",
        choices=("audit", "qualify", "fork", "trace", "repeat"),
        required=True,
    )
    parser.add_argument("--trial", type=int, default=0)
    args = parser.parse_args()
    config = json.loads((args.root / "paths.json").read_text())
    check_runtime(args.root, config)
    if args.stage == "audit":
        audit(args.root, config)
        return
    receipt = json.loads((args.root / "SOURCE_AUDITED.json").read_text())
    if receipt["config_sha256"] != digest(args.root / "paths.json") or receipt[
        "runtime_sha256"
    ] != digest(args.root / "runtime-contract.json"):
        raise ValueError("Source audit does not match frozen configuration/runtime")
    from samudra.rust_data import create_rust_io_runtime

    create_rust_io_runtime(4)
    print(
        json.dumps(
            dict(
                event="stability_runtime",
                gpu=torch.cuda.get_device_name(),
                producer=config["producer"],
                stage=args.stage,
                arm=args.arm,
            )
        ),
        flush=True,
    )
    try:
        if args.stage == "qualify":
            for stage, marker in (
                ("fit", f"fit-{args.arm}/QUALIFIED.json"),
                ("probe", f"probe-{args.arm}/THROUGHPUT.json"),
            ):
                if not (args.root / marker).exists():
                    directory = (args.root / marker).parent
                    if directory.exists():
                        attempts = args.root / "attempts"
                        attempts.mkdir(exist_ok=True)
                        directory.rename(
                            attempts / f"{directory.name}-{time.time_ns()}"
                        )
                    command(
                        "scripts.run_early_fine_rmse_wave",
                        [
                            "--root",
                            args.root,
                            "--arm",
                            args.arm,
                            "--stage",
                            stage,
                        ],
                    )

        elif args.stage == "fork":
            fork(args.root, args.arm, config)
        elif args.stage == "trace":
            trace(args.root, args.arm, args.trial)
        else:
            repeat(args.root, args.arm, args.trial)
    except InterruptedError:
        # EarlyPilot saved an exact restart checkpoint before raising.
        raise SystemExit(75) from None


if __name__ == "__main__":
    main()
