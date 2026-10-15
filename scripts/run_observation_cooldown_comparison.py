#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Matched data/loss/EMA cooldown branches from the completed 4k S0 states."""

import argparse
import json
import time
from pathlib import Path

import torch

from samudra.experiments.early_fine_training import deterministic_replay
from samudra.experiments.missingness import corrupt_sample
from samudra.experiments.observation_joint import assert_exact_state, cpu_copy
from samudra.experiments.observation_pilot import atomic_json, digest
from samudra.experiments.task_schedule import TaskSchedule
from scripts.run_observation_stability import ARMS, check_runtime, close, make_pilot
from scripts.run_observation_stability_screen import (
    ScreenPilot,
    WeightEMA,
    evaluate_pair,
    restore,
)

BRANCHES = ("A", "B", "C", "D")


class CooldownSchedule:
    """Preserve the full source schedule, append a fixed 7:1 or observation tail."""

    total = 5000
    observation_finish = 0

    def __init__(self, branch):
        if branch not in BRANCHES:
            raise ValueError("Unknown cooldown branch")
        self.branch = branch
        self.prefix = TaskSchedule(2000, 2000, "mixed")
        self.om4_updates = 2125 if branch == "A" else 2000
        self.observation_updates = self.total - self.om4_updates

    def counts(self, completed):
        if not 0 <= completed <= self.total:
            raise ValueError("Update outside cooldown schedule")
        if completed <= 4000:
            return self.prefix.counts(completed)
        extra = completed - 4000
        om4 = extra // 8 if self.branch == "A" else 0
        return dict(om4=2000 + om4, observation=2000 + extra - om4)

    def task(self, index):
        if not 0 <= index < self.total:
            raise ValueError("Update outside cooldown schedule")
        return (
            "om4"
            if self.counts(index + 1)["om4"] > self.counts(index)["om4"]
            else "observation"
        )


class CooldownPilot(ScreenPilot):
    def __init__(self, args, branch):
        self.branch = branch
        super().__init__(args, branch)

    def make_schedule(self, args):
        if (args.om4_updates, args.joint_steps, args.ordering) != (2000, 2000, "mixed"):
            raise ValueError("Unexpected source schedule")
        return CooldownSchedule(self.branch)

    def contract(self):
        return dict(
            policy=self.branch,
            start=4000,
            end=5000,
            ema_half_life=100,
            ema_initialization="retain" if self.branch == "D" else "raw",
            buffers="copy raw buffers each update",
            mixture="7 observation then 1 OM4"
            if self.branch == "A"
            else "observation only",
            objective="forecast only, same input corruption"
            if self.branch == "C"
            else "unchanged",
            lr_start=1e-4,
            lr_end=1e-6,
            producer=self.manifest["code_commit"],
        )

    def task_learning_rate(self, task):
        update = self.completed + 1
        if not 4001 <= update <= 5000:
            raise ValueError("Wrong cooldown update")
        return 1e-4 * (1 - 0.99 * (update - 4000) / 1000)

    def objective(self, sample, phase):
        if self.branch != "C":
            return super().objective(sample, phase)
        if phase != "joint":
            raise ValueError("Forecast-only branch cannot train reconstruction phase")
        # Keep the same masked inputs as B; remove only the two loss terms.
        sample = corrupt_sample(sample, sample.get("mask_seed", self.args.seed))
        prediction, _ = self.model.forecast(*self.arguments(sample))
        return self.data.forecast_loss(prediction, sample)

    def activate_source(self, source):
        saved = torch.load(self.resume, map_location="cpu", weights_only=False)
        self.ema = WeightEMA(self.model)
        if "screen" in saved:
            if saved["screen"] != self.contract():
                raise ValueError("Cooldown resume contract changed")
            for key, value in saved["ema"].items():
                self.ema.state[key].copy_(value)
            self.ema.updates = saved["ema_updates"]
        else:
            if self.completed != 4000:
                raise ValueError("Missing cooldown EMA state")
            parent = torch.load(source, map_location="cpu", weights_only=False)
            if (
                parent["global_step"] != 4000
                or parent["ema_updates"] != 1000
                or parent["screen"]["policy"] != "S0"
            ):
                raise ValueError("Source is not completed S0 with retained EMA")
            assert_exact_state(parent["model"], self.model.state_dict())
            assert_exact_state(parent["optimizer"], self.optimizer.state_dict())
            if self.branch == "D":
                for key, value in parent["ema"].items():
                    self.ema.state[key].copy_(value)
                self.ema.updates = parent["ema_updates"]
            atomic_json(
                dict(
                    contract=self.contract(),
                    source=str(source),
                    source_sha256=digest(source),
                    source_ema_updates=parent["ema_updates"],
                    initial_ema_updates=self.ema.updates,
                    optimizer_and_raw_exact=True,
                ),
                self.out / "COOLDOWN_MIGRATION.json",
            )
        expected = self.completed - 4000 + (1000 if self.branch == "D" else 0)
        if self.ema.updates != expected:
            raise ValueError("EMA count differs from cooldown contract")
        self.checkpoint(self.resume)


def run(root, arm, branch, qualify):
    config = json.loads((root / "paths.json").read_text())
    check_runtime(root, config)
    audit = json.loads((root / "SOURCE_AUDITED.json").read_text())
    if audit["config_sha256"] != digest(root / "paths.json") or audit[
        "runtime_sha256"
    ] != digest(root / "runtime-contract.json"):
        raise ValueError("Audit drift")
    source = config["cooldown_sources"][arm]
    qualified = root / arm / ("cooldown-qualify-" + branch)
    marker = qualified / "COOLDOWN_QUALIFIED.json"
    out = qualified if qualify else root / arm / branch
    if not qualify:
        gate = json.loads(marker.read_text())
        if (
            gate["producer"] != config["producer"]
            or gate["source_sha256"] != source["sha256"]
            or gate["branch"] != branch
            or not gate["bitwise_replay_exact"]
        ):
            raise ValueError("Cooldown qualification differs")
    if (marker if qualify else out / "COMPLETE.json").exists():
        return
    if qualify and out.exists():
        out.rename(out.with_name(out.name + "-failed-" + str(time.time_ns())))
    pilot = make_pilot(
        root,
        arm,
        out,
        Path(source["checkpoint"]),
        source["sha256"],
        pilot_class=lambda args: CooldownPilot(args, branch),
    )
    try:
        pilot.activate_source(Path(source["checkpoint"]))
        if qualify:
            initial = torch.load(pilot.resume, map_location="cpu", weights_only=False)
            expected = None
            for trial in range(2):
                restore(pilot, initial)
                with deterministic_replay():
                    for _ in range(16):
                        pilot.train_update()
                pilot.checkpoint(out / f"replay-{trial}.pt")
                saved = torch.load(
                    out / f"replay-{trial}.pt", map_location="cpu", weights_only=False
                )
                keys = (
                    "model",
                    "optimizer",
                    "global_step",
                    "task_counts",
                    "rng_cpu",
                    "rng_cuda",
                    "rng_numpy",
                    "screen",
                    "ema",
                    "ema_updates",
                )
                observed = {key: saved[key] for key in keys}
                if expected is None:
                    expected = cpu_copy(observed)
                else:
                    assert_exact_state(observed, expected)
            raw = cpu_copy(pilot.model.state_dict())
            evaluate_pair(pilot)
            assert_exact_state(raw, pilot.model.state_dict())
            atomic_json(
                dict(
                    producer=config["producer"],
                    source_sha256=source["sha256"],
                    branch=branch,
                    bitwise_replay_exact=True,
                    raw_restored_after_ema=True,
                    diagnostic_weights_only=True,
                    qualification_updates=16,
                    task_counts=pilot.schedule.counts(4016),
                ),
                marker,
            )
        else:
            if pilot.completed == 4000:
                evaluate_pair(pilot)
            while pilot.completed < 5000:
                pilot.train_update()
                if pilot.completed % 100 == 0:
                    evaluate_pair(pilot)
                    pilot.checkpoint(pilot.resume)
            atomic_json(
                dict(
                    global_step=pilot.completed,
                    branch=branch,
                    source_sha256=source["sha256"],
                    sha256=digest(pilot.resume),
                    ema_updates=pilot.ema.updates,
                    cooldown_updates=1000,
                    task_counts=pilot.schedule.counts(pilot.completed),
                ),
                out / "COMPLETE.json",
            )
    finally:
        close(pilot)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--branch", choices=BRANCHES, required=True)
    parser.add_argument("--qualify", action="store_true")
    args = parser.parse_args()
    from samudra.rust_data import create_rust_io_runtime

    create_rust_io_runtime(4)
    try:
        run(args.root, args.arm, args.branch, args.qualify)
    except InterruptedError:
        raise SystemExit(75) from None


if __name__ == "__main__":
    main()
