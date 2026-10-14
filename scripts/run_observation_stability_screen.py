#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Paired Stage-2 continuations, with atomic raw/EMA/optimizer restart state."""

import argparse
import contextlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from samudra.experiments.early_fine_training import EarlyPilot, deterministic_replay
from samudra.experiments.observation_joint import assert_exact_state, cpu_copy
from samudra.experiments.observation_pilot import atomic_json, atomic_torch, digest
from scripts.run_observation_stability import (
    ARMS,
    check_runtime,
    close,
    make_pilot,
    preserve_execution,
    validation,
)

POLICIES = ("S0", "S0-repeat", "S1", "S2")


def learning_rate(policy, step):
    if policy not in POLICIES or not 3001 <= step <= 4000:
        raise ValueError("Invalid screen policy/update")
    if policy == "S1":
        return 3e-5
    if policy == "S2":
        return 1e-4 * (1 - 0.99 * (step - 3000) / 1000)
    return 1e-4


class WeightEMA:
    """Average parameters; copy buffers exactly from the current raw model."""

    def __init__(self, model):
        self.state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.parameters = set(dict(model.named_parameters()))
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        decay = 2 ** (-1 / 100)
        for key, value in model.state_dict().items():
            if key in self.parameters:
                self.state[key].mul_(decay).add_(value, alpha=1 - decay)
            else:
                self.state[key].copy_(value)
        self.updates += 1

    @contextlib.contextmanager
    def applied(self, model):
        raw = {k: v.detach().clone() for k, v in model.state_dict().items()}
        try:
            model.load_state_dict(self.state, strict=True)
            yield
        finally:
            model.load_state_dict(raw, strict=True)


class ScreenPilot(EarlyPilot):
    def __init__(self, args, policy):
        self.policy = policy
        self.ema = None
        self.in_update = False
        super().__init__(args)
        if args.core_lr != 1e-4 or args.om4_lr != 1e-4 or args.warmup_steps:
            raise ValueError("Unexpected source learning-rate contract")
        if getattr(args, "cooldown_start", None) is not None:
            raise ValueError("Screen policy must not stack another cooldown")

    def contract(self):
        return dict(
            policy=self.policy,
            start=3000,
            end=4000,
            ema_half_life=100,
            buffers="copy raw buffers each update",
            producer=self.manifest["code_commit"],
        )

    def activate(self):
        saved = torch.load(self.resume, map_location="cpu", weights_only=False)
        self.ema = WeightEMA(self.model)
        if "screen" in saved:
            if saved["screen"] != self.contract():
                raise ValueError("Screen resume contract changed")
            for key, value in saved["ema"].items():
                self.ema.state[key].copy_(value)
            self.ema.updates = saved["ema_updates"]
        elif self.completed != 3000:
            raise ValueError("Missing EMA restart state")
        if self.ema.updates != self.completed - 3000:
            raise ValueError("EMA update count differs from optimizer count")
        self.checkpoint(self.resume)

    def task_learning_rate(self, task):
        return learning_rate(self.policy, self.completed + 1)

    def checkpoint(self, path, model_only=False):
        if self.in_update:
            return  # Save only after EMA has consumed the same optimizer update.
        if self.ema is None:
            return super().checkpoint(path, model_only)
        atomic_torch(
            dict(
                model=self.model.state_dict(),
                optimizer=None if model_only else self.optimizer.state_dict(),
                manifest=self.manifest,
                global_step=self.completed,
                step=self.schedule.counts(self.completed)["observation"],
                phase="joint",
                task_counts=self.schedule.counts(self.completed),
                rng_cpu=torch.get_rng_state(),
                rng_cuda=torch.cuda.get_rng_state(),
                rng_numpy=np.random.get_state(),
                elapsed_seconds=self.elapsed_seconds + time.monotonic() - self.started,
                screen=self.contract(),
                ema=self.ema.state,
                ema_updates=self.ema.updates,
            ),
            path,
        )

    def train_update(self):
        if self.stop_requested:
            self.checkpoint(self.resume)
            raise InterruptedError("Screen checkpoint saved before requeue")
        self.in_update = True
        try:
            super().train_update()
            assert self.ema is not None
            self.ema.update(self.model)
        finally:
            self.in_update = False
        if self.completed % 10 == 0:
            self.checkpoint(self.resume)


def restore(pilot, saved):
    if saved["screen"] != pilot.contract():
        raise ValueError("Replay policy changed")
    pilot.model.load_state_dict(saved["model"], strict=True)
    pilot.optimizer.load_state_dict(saved["optimizer"])
    pilot.completed = saved["global_step"]
    for key, value in saved["ema"].items():
        pilot.ema.state[key].copy_(value)
    pilot.ema.updates = saved["ema_updates"]
    torch.set_rng_state(saved["rng_cpu"])
    torch.cuda.set_rng_state(saved["rng_cuda"])
    np.random.set_state(saved["rng_numpy"])


def evaluate_pair(pilot):
    step = pilot.completed
    raw = validation(pilot, f"raw-{step:05d}")
    with preserve_execution(pilot.model), pilot.ema.applied(pilot.model):
        ema = validation(pilot, f"ema-{step:05d}")
    pilot.emit(
        dict(
            event="screen_validation",
            global_step=step,
            raw_rmse_blend=raw["score"],
            ema_rmse_blend=ema["score"],
            policy=pilot.policy,
        )
    )
    return raw, ema


def run(root, arm, policy, qualify):
    config = json.loads((root / "paths.json").read_text())
    check_runtime(root, config)
    audit = json.loads((root / "SOURCE_AUDITED.json").read_text())
    if audit["config_sha256"] != digest(root / "paths.json") or audit[
        "runtime_sha256"
    ] != digest(root / "runtime-contract.json"):
        raise ValueError("Source/runtime audit drift")
    source = config["screen_sources"][arm]
    qualification = root / arm / ("screen-qualify-" + policy)
    marker = qualification / "SCREEN_QUALIFIED.json"
    if not qualify:
        gate = json.loads(marker.read_text())
        if (
            gate["producer"] != config["producer"]
            or gate["source_sha256"] != source["sha256"]
            or gate["policy"] != policy
        ):
            raise ValueError("Screen qualification lineage mismatch")
        if not gate["bitwise_replay_exact"]:
            raise ValueError("Screen replay failed")
    out = qualification if qualify else root / arm / policy
    if (marker if qualify else out / "COMPLETE.json").exists():
        return
    # A failed qualification is never resumed as if it were an independent proof.
    if qualify and out.exists():
        out.rename(out.with_name(out.name + "-failed-" + str(time.time_ns())))
    pilot = make_pilot(
        root,
        arm,
        out,
        Path(source["checkpoint"]),
        source["sha256"],
        pilot_class=lambda args: ScreenPilot(args, policy),
    )
    try:
        pilot.activate()
        if qualify:
            initial = torch.load(pilot.resume, map_location="cpu", weights_only=False)
            expected = None
            for trial in range(2):
                restore(pilot, initial)
                with deterministic_replay():
                    for _ in range(8):
                        pilot.train_update()
                pilot.checkpoint(out / f"replay-{trial}.pt")
                observed = torch.load(
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
                if expected is None:
                    expected = cpu_copy({k: observed[k] for k in keys})
                    # Next execution explicitly reloads the serialized initial state.
                    initial = torch.load(
                        pilot.resume, map_location="cpu", weights_only=False
                    )
                else:
                    for key in keys:
                        assert_exact_state(expected[key], observed[key])
            raw_before = cpu_copy(pilot.model.state_dict())
            evaluate_pair(pilot)
            assert_exact_state(raw_before, pilot.model.state_dict())
            atomic_json(
                dict(
                    producer=config["producer"],
                    policy=policy,
                    source_sha256=source["sha256"],
                    bitwise_replay_exact=True,
                    optimizer_and_ema_atomic=True,
                    raw_restored_after_ema=True,
                    diagnostic_weights_only=True,
                ),
                marker,
            )
        else:
            if pilot.completed == 3000:
                evaluate_pair(pilot)
            while pilot.completed < 4000:
                pilot.train_update()
                if pilot.completed % 100 == 0:
                    evaluate_pair(pilot)
                    pilot.checkpoint(pilot.resume)
            atomic_json(
                dict(
                    global_step=pilot.completed,
                    sha256=digest(pilot.resume),
                    source_sha256=source["sha256"],
                    policy=policy,
                    ema_updates=pilot.ema.updates,
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
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--qualify", action="store_true")
    args = parser.parse_args()
    from samudra.rust_data import create_rust_io_runtime

    create_rust_io_runtime(4)
    try:
        run(args.root, args.arm, args.policy, args.qualify)
    except InterruptedError:
        raise SystemExit(75) from None


if __name__ == "__main__":
    main()
