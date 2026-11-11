# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Explicit, provenance-checked continuation distinct from strict job resume."""

import hashlib
import math
from pathlib import Path

import torch

from .multitask import learning_rate_factor

# These controls change runtime/recording or the explicitly replaced schedule.
# Data, sampling microbatch, accumulation, objective, model and EMA stay exact.
CONTINUATION_OVERRIDES = frozenset(
    {
        "continuation_parent",
        "continuation_parent_sha256",
        "continuation_schedule",
        "max_updates",
        "schedule_updates",
        "validate_at",
        "validate_updates",
        "save_seconds",
        "log_every",
        "max_allocated_seconds",
        "wandb",
        "resume",
        "physical_microbatch",
        "prefetch_microbatches",
    }
)


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def continuation_learning_rate(step, schedule):
    expected = {
        "start_step",
        "rewarm_updates",
        "start_lr",
        "peak_lr",
        "end_lr",
        "end_step",
    }
    if set(schedule) != expected:
        raise ValueError("Continuation schedule requires explicit endpoints and rewarm")
    start, rewarm, end = (
        schedule["start_step"],
        schedule["rewarm_updates"],
        schedule["end_step"],
    )
    if (
        any(not isinstance(x, int) or isinstance(x, bool) for x in (start, rewarm, end))
        or not 0 <= start < start + rewarm < end
        or not start <= step <= end
    ):
        raise ValueError("Invalid continuation update interval")
    low, peak, final = (schedule[k] for k in ("start_lr", "peak_lr", "end_lr"))
    if not all(math.isfinite(x) and x > 0 for x in (low, peak, final)) or peak < max(
        low, final
    ):
        raise ValueError("Invalid continuation learning rates")
    if step <= start + rewarm:
        return low + (peak - low) * (step - start) / rewarm
    fraction = (step - start - rewarm) / (end - start - rewarm)
    return final + (peak - final) * (1 + math.cos(math.pi * fraction)) / 2


def comparison_learning_rate(step, config):
    if "continuation_parent" in config:
        return continuation_learning_rate(step, config["continuation_schedule"])
    return config["learning_rate"] * learning_rate_factor(
        step, config["warmup"], config.get("schedule_updates", config["max_updates"])
    )


def validate_continuation(state, config, source_receipt):
    if state["source"] != source_receipt:
        raise ValueError("Data provenance changed for continuation")
    old = state["config"]
    changed = {
        k
        for k in old.keys() | config.keys()
        if (k not in old or k not in config or old[k] != config[k])
        and k not in CONTINUATION_OVERRIDES
    }
    if changed:
        raise ValueError(
            f"Continuation changes protected configuration: {sorted(changed)}"
        )
    if "continuation_parent" in old:
        raise ValueError("Nested continuation is not supported; resume its own output")
    schedule = config["continuation_schedule"]
    start = state["step"]
    rate = continuation_learning_rate(start, schedule)
    if schedule["start_step"] != start or schedule["end_step"] != config["max_updates"]:
        raise ValueError("Continuation endpoints differ from parent/update budget")
    if config.get("schedule_updates", config["max_updates"]) != schedule["end_step"]:
        raise ValueError("Continuation schedule budget is inconsistent")
    if int(state["ema"]["num_updates"]) != start:
        raise ValueError("Parent EMA count does not match checkpoint step")
    if len(state["rank_states"]) != config["world_size"]:
        raise ValueError("Continuation rank states do not match world size")
    if not state["optimizer"]["state"]:
        raise ValueError("Continuation requires populated optimizer state")
    for group in state["optimizer"]["param_groups"]:
        if not math.isclose(float(group["lr"]), rate, rel_tol=1e-8, abs_tol=1e-12):
            raise ValueError("Continuation start LR differs from saved optimizer LR")
    for value in state["optimizer"]["state"].values():
        if int(value["step"]) != start:
            raise ValueError("Parent optimizer count does not match checkpoint step")
    physical = config.get("physical_microbatch", config["microbatch"])
    logical = config["microbatch"]
    if (
        not isinstance(physical, int)
        or isinstance(physical, bool)
        or physical < logical
        or physical % logical
        or config["accumulation"] % (physical // logical)
    ):
        raise ValueError("Physical microbatch must group whole original sampling calls")
    return {
        "parent_checkpoint": str(Path(config["continuation_parent"]).resolve()),
        "parent_sha256": config["continuation_parent_sha256"],
        "parent_step": start,
        "schedule": schedule,
        "changed_controls": {
            k: {"parent": old.get(k), "continuation": config.get(k)}
            for k in sorted(old.keys() | config.keys())
            if k not in old or k not in config or old[k] != config[k]
        },
    }


def load_comparison_checkpoint(config, output, source_receipt, restart=0):
    latest = Path(output) / "latest.pt"
    if latest.exists():
        if not (restart or config.get("resume", False)):
            raise FileExistsError(latest)
        state = torch.load(latest, map_location="cpu", mmap=True, weights_only=False)
        if state["config"] != config or state["source"] != source_receipt:
            raise ValueError("Configuration or data provenance changed on resume")
        return state, False, state.get("continuation")
    if "continuation_parent" not in config:
        if any(k.startswith("continuation_") for k in config):
            raise ValueError("Continuation settings require a parent checkpoint")
        return None, False, None
    parent = Path(config["continuation_parent"])
    if parent.resolve().is_relative_to(Path(output).resolve()):
        raise ValueError("Continuation parent must be outside the new output directory")
    if checkpoint_sha256(parent) != config["continuation_parent_sha256"]:
        raise ValueError("Continuation parent checkpoint checksum mismatch")
    state = torch.load(parent, map_location="cpu", mmap=True, weights_only=False)
    receipt = validate_continuation(state, config, source_receipt)
    return state, True, receipt


def restore_comparison_state(model, optimizer, rng, state, rank, device):
    """Restore all trained and sampling state without changing the parent object."""
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    saved = state["rank_states"][rank]
    rng.bit_generator.state = saved["rng"]
    torch.set_rng_state(saved["torch_rng"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(saved["cuda_rng"], device)
    return state["step"], dict(state["best"]), saved["stream"]
