#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Verify the paired eight-GPU fresh/resume smoke before production submission."""

import argparse
import json
import math
from pathlib import Path

import torch

from samudra.experiments.diffusion_direct import checkpoint_processor_prediction
from samudra.experiments.diffusion_scratch_distributed import (
    learning_rate,
    owned_replay_slots,
    validate_rank_states,
)
from samudra.experiments.observation_pilot import atomic_json, digest
from samudra.experiments.task_schedule import TaskSchedule

TOTAL = 20320
WORLD = 8
MARKER = "processor.direct_prediction_version"


def read(path):
    return json.loads(path.read_text())


def verify_events(events, schedule):
    updates = [event for event in events if event.get("event") == "update"]
    if [event["step"] for event in updates] != list(range(1, 33)):
        raise ValueError("Smoke updates must be continuous exactly 1 through 32")
    stops = [event for event in events if event.get("event") == "invocation_finished"]
    if [event["step"] for event in stops] != [16, 32]:
        raise ValueError("Expected two actual invocations stopping at 16 and 32")
    boundary = events.index(stops[0])
    if [e["step"] for e in events[:boundary] if e.get("event") == "update"] != list(
        range(1, 17)
    ):
        raise ValueError("First invocation must stop after update 16")
    for event in updates:
        step = event["step"]
        task = schedule.task(step - 1)
        if event["task"] != task or event["counts"] != schedule.counts(step):
            raise ValueError("Smoke task exposure differs from production schedule")
        if (
            event["task_update"] != schedule.counts(step)[task]
            or event["world_size"] != WORLD
        ):
            raise ValueError("Wrong task counter or distributed world size")
        if event["learning_rate"] != learning_rate(step - 1, TOTAL):
            raise ValueError("Smoke learning rate differs from production schedule")
        values = [
            event[key]
            for key in (
                "loss",
                "forecast",
                "completion",
                "reconstruction",
                "gradient_norm",
                "seconds",
            )
        ]
        gradients = event["gradient_norms"]
        if set(gradients) != {"initializer", "processor", "decoder"}:
            raise ValueError("Incomplete module gradient diagnostics")
        if not all(math.isfinite(v) for v in [*values, *gradients.values()]):
            raise ValueError("Nonfinite smoke losses/gradients")
        if (
            event["gradient_norm"] <= 0
            or event["seconds"] <= 0
            or any(v <= 0 for v in gradients.values())
        ):
            raise ValueError("Missing module gradients or invalid timing")
    for event in stops:
        if (
            event["complete"]
            or event["counts"] != schedule.counts(event["step"])
            or event["world_size"] != WORLD
        ):
            raise ValueError("Invalid smoke pause receipt")
    return updates


def verify_arm(root, arm, code_id):
    run = root / "smoke" / arm
    protocol = read(run / "protocol.json")
    required = dict(
        producer=code_id,
        arm=arm,
        processor_prediction=arm,
        updates=TOTAL,
        world_size=WORLD,
        effective_batch=8,
        seed=1729,
        compile_decoder=True,
        autocast="bfloat16",
        initialization="fresh same-seed random weights",
        sampling_steps=32,
        members_train=2,
        task_schedule="quadratic mixed",
        cooldown_start=17780,
        minimum_lr=1e-6,
        latent_bound_weight=0.1 if arm == "residual" else 0.0,
    )
    if any(protocol.get(key) != value for key, value in required.items()):
        raise ValueError(f"Wrong scientific/producer qualification protocol: {arm}")
    schedule = TaskSchedule(10160, 10160, "mixed")
    events = [
        json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()
    ]
    updates = verify_events(events, schedule)
    last = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    if (
        last["step"] != 32
        or last["counts"] != schedule.counts(32)
        or last["intervention"] != protocol
    ):
        raise ValueError("Final smoke checkpoint exposure/protocol differs")
    if (
        checkpoint_processor_prediction(last) != arm
        or last["contract"]["producer"] != code_id
    ):
        raise ValueError("Final smoke checkpoint architecture/producer differs")
    if not last["optimizer"]["state"]:
        raise ValueError("Missing optimizer moments in resumed checkpoint")
    validate_rank_states(last["rank_states"], WORLD, require_populated=True)
    for state in last["rank_states"]:
        if not all(
            key in state for key in ("cpu_rng", "cuda_rng", "numpy_rng", "python_rng")
        ):
            raise ValueError("Incomplete rank RNG checkpoint")
        if state["cuda_rng"] is None:
            raise ValueError("Missing CUDA RNG checkpoint")
    for filename in ("state.json", "PAUSED.json"):
        state = read(run / filename)
        if (
            state["step"] != 32
            or state["counts"] != schedule.counts(32)
            or state["complete"]
        ):
            raise ValueError("State/pause output disagrees with resumed checkpoint")
    om4_steps = [event["step"] for event in updates if event["task"] == "om4"]
    evidence = [
        "protocol.json",
        "events.jsonl",
        "last.pt",
        "state.json",
        "PAUSED.json",
        "step-00000.pt",
    ]
    for rank in range(WORLD):
        slots = owned_replay_slots(rank, WORLD)
        if not slots:
            continue
        filename = f"replay-rank-{rank}.jsonl"
        records = [
            json.loads(line) for line in (run / filename).read_text().splitlines()
        ]
        if [record["step"] for record in records] != om4_steps:
            raise ValueError("Missing/duplicated replay updates across restart")
        if any([r["slot"] for r in record["records"]] != slots for record in records):
            raise ValueError("Replay log ownership differs")
        evidence.append(filename)
    del last
    initial = torch.load(run / "step-00000.pt", map_location="cpu", weights_only=False)
    if (
        initial["step"] != 0
        or initial["counts"] != {"om4": 0, "observation": 0}
        or initial["intervention"] != protocol
    ):
        raise ValueError("Wrong fresh initialization snapshot")
    if checkpoint_processor_prediction(initial) != arm:
        raise ValueError("Fresh snapshot architecture differs")
    return initial["model"], dict(
        counts=schedule.counts(32),
        step=32,
        world_size=WORLD,
        receipts={filename: digest(run / filename) for filename in evidence},
        task_update_seconds={
            task: sum(e["seconds"] for e in updates if e["task"] == task)
            / sum(e["task"] == task for e in updates)
            for task in ("om4", "observation")
        },
    )


def verify_initial_pair(direct, residual):
    direct = {key: value for key, value in direct.items() if key != MARKER}
    if set(direct) != set(residual):
        raise ValueError("Fresh paired models have different parameter/buffer keys")
    for key in direct:
        if (
            not torch.equal(direct[key], residual[key])
            or not torch.isfinite(direct[key]).all()
        ):
            raise ValueError(
                f"Fresh paired model tensor differs or is nonfinite: {key}"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--code-id", required=True)
    args = parser.parse_args()
    direct, direct_receipt = verify_arm(args.root, "direct", args.code_id)
    residual, residual_receipt = verify_arm(args.root, "residual", args.code_id)
    verify_initial_pair(direct, residual)
    result = dict(
        qualified=True,
        producer=args.code_id,
        world_size=WORLD,
        production_updates=TOTAL,
        completed_smoke_updates=32,
        resume_boundary=16,
        same_initial_tensors=True,
        arms=dict(direct=direct_receipt, residual=residual_receipt),
        scope="Fresh paired initialization; actual 16-to-32 resume; finite full-objective gradients and complete replay shards. No claim of scientific convergence or uninterrupted/resumed bitwise equality.",
    )
    atomic_json(result, args.root / "QUALIFIED.json")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
