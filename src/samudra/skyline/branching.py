# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Explicit checkpoint branches with preserved scientific configuration."""

import hashlib
import math
from pathlib import Path

import torch

from .multitask import learning_rate_factor

BRANCH_OVERRIDES = {
    "branch_checkpoint",
    "learning_rate",
    "learning_rate_schedule",
    "max_updates",
    "validate_at",
    "validate_updates",
    "qualification",
    "wandb",
    "log_every",
}


def load_branch(config, source):
    specification = config["branch_checkpoint"]
    path = Path(specification["path"])
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != specification["sha256"]:
        raise ValueError("Branch checkpoint checksum differs")
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    differences = {
        key
        for key in state["config"].keys() | config.keys()
        if state["config"].get(key) != config.get(key)
    }
    if differences - BRANCH_OVERRIDES:
        raise ValueError(f"Branch changed scientific configuration: {differences}")
    if state["source"] != source:
        raise ValueError("Branch data provenance differs")
    if state["step"] != specification["step"]:
        raise ValueError("Branch checkpoint update differs")
    if len(state["rank_states"]) != config["world_size"]:
        raise ValueError("Branch rank states differ from world size")
    if config["max_updates"] <= state["step"]:
        raise ValueError("Branch must add optimizer updates")
    return state


def load_resume(config, source, latest, authorized):
    """Resume an existing output only with explicit or scheduler authorization."""
    if not authorized:
        raise FileExistsError(latest)
    state = torch.load(latest, map_location="cpu", mmap=True, weights_only=False)
    if state["config"] != config or state["source"] != source:
        raise ValueError("Configuration or data provenance changed on resume")
    return state


def scheduled_learning_rate(config, step):
    schedule = config.get("learning_rate_schedule", "cosine")
    if schedule == "constant":
        return config["learning_rate"]
    if schedule == "warmup_stable_cosine":
        warmup = config["warmup"]
        cooldown_start = config["cooldown_start"]
        end = config.get("schedule_updates", config["max_updates"])
        floor = config["final_learning_rate"]
        peak = config["learning_rate"]
        if not (0 < warmup < cooldown_start < end and 1 <= step <= end):
            raise ValueError("Invalid warmup/stable/cooldown schedule")
        if not 0 <= floor <= peak:
            raise ValueError("Final learning rate must lie between zero and peak")
        if step <= warmup:
            return peak * step / warmup
        fraction = max(0.0, (step - cooldown_start) / (end - cooldown_start))
        return floor + (peak - floor) * (1 + math.cos(math.pi * fraction)) / 2
    if schedule != "cosine":
        raise ValueError(f"Unknown learning-rate schedule: {schedule}")
    return config["learning_rate"] * learning_rate_factor(
        step,
        config["warmup"],
        config.get("schedule_updates", config["max_updates"]),
    )
